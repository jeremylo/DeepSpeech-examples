import logging
from datetime import datetime
import collections
import queue
import os
import deepspeech
import numpy as np
import pyaudio
import wave
from scipy.signal.filter_design import butter
from scipy.signal.signaltools import lfilter, lfilter_zi
import webrtcvad
from halo import Halo
from scipy import signal

logging.basicConfig(level=20)


class VoiceAudioService:

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    sample_rate = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, aggressiveness=2, device=None, input_rate=None, file=None):
        def callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            self.buffer_queue.put(in_data)
            return (None, pyaudio.paContinue)

        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.block_size = int(self.sample_rate /
                              float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(
            self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

        self.vad = webrtcvad.Vad(aggressiveness)

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so resample from input_rate to sample_rate here for webrtcvad and deepspeech.

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.frombuffer(data, dtype=np.int16)
        resample_size = int(len(data16) * self.sample_rate / self.input_rate)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    filter_enabled = True
    lowpass_frequency = 60
    highpass_frequency = 6000

    def filter(self, data):
        if not self.filter_enabled:
            return data

        data16 = np.frombuffer(data, dtype=np.int16)

        nyquist_frequency = 0.5 * self.sample_rate
        b, a = butter(2, [
            self.lowpass_frequency / nyquist_frequency,
            self.highpass_frequency / nyquist_frequency
        ], btype='bandpass')

        filtered, _ = lfilter(b, a, data16, axis=0, zi=lfilter_zi(b, a))

        return np.array(filtered, dtype=np.int16).tobytes()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.filter(self.buffer_queue.get())

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)  # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

    def frames(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.sample_rate:
            while True:
                yield self.read()
        else:
            while True:
                yield self.resample(self.read())

    def utterances(self, padding_ms=300, ratio=0.75):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """

        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        try:
            for frame in self.frames():
                if len(frame) < 640:
                    return

                is_speech = self.vad.is_speech(frame, self.sample_rate)

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len(
                        [f for f, speech in ring_buffer if speech])
                    if num_voiced > ratio * ring_buffer.maxlen:
                        triggered = True
                        for f, s in ring_buffer:
                            yield f
                        ring_buffer.clear()

                else:
                    yield frame
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len(
                        [f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > ratio * ring_buffer.maxlen:
                        triggered = False
                        yield None
                        ring_buffer.clear()
        except KeyboardInterrupt:
            pass
        finally:
            self.destroy()


def init_deepspeech(model, scorer):
    print("Initialising model: %s" % model)
    logging.info("Initialising model: %s", model)

    if os.path.isdir(model):
        model = os.path.join(model, 'output_graph.pb')
        scorer = os.path.join(model, scorer)

    model = deepspeech.Model(model)
    if scorer:
        logging.info("Scorer: %s", scorer)
        model.enableExternalScorer(scorer)

    return model


def transcribe(model, vas, nospinner, savewav):
    if savewav:
        os.makedirs(savewav, exist_ok=True)

    spinner = None
    if not nospinner:
        spinner = Halo(spinner='line')

    stream_context = model.createStream()
    wav_data = bytearray()
    for utterance in vas.utterances():
        if utterance is not None:
            if spinner:
                spinner.start()

            logging.debug("start utterance")
            stream_context.feedAudioContent(np.frombuffer(utterance, np.int16))
            if savewav:
                wav_data.extend(utterance)
        else:
            if spinner:
                spinner.stop()

            logging.debug("end utterence")

            if savewav:
                vas.write_wav(os.path.join(savewav, datetime.now().strftime(
                    "savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()

            yield stream_context.finishStream()

            stream_context = model.createStream()


def main(args):
    # Load the DeepSpeech model
    model = init_deepspeech(args.model, args.scorer)

    # Start audio
    vas = VoiceAudioService(aggressiveness=args.aggressiveness,
                            device=args.device,
                            input_rate=args.rate,
                            file=args.file)

    print("Listening (Ctrl-C to exit).")

    # Stream from microphone to DeepSpeech using VAD
    for text in transcribe(model, vas, args.nospinner, args.savewav):
        if text:
            print("Recognised: %s" % text)


if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(
        description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-a', '--aggressiveness', type=int, default=2,
                        help="Set the voice activity detection aggressiveness: an integer between 0 and 3, with 0 being the least aggressive at filtering out non-speech and 3 the most aggressive. Default: 2.")

    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")

    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")

    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")

    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")

    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")

    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")

    main(parser.parse_args())
