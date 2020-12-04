import logging
from datetime import datetime
import collections
import queue
import os
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
import halo
from scipy import signal

logging.basicConfig(level=20)


class VoiceAudioService:

    format = pyaudio.paInt16

    # Network/VAD rate-space
    channels = 1
    sample_rate = 16000
    blocks_per_second = 50

    block_size = sample_rate // blocks_per_second

    def __init__(self, aggressiveness=2, device=None, input_rate=None, file=None, filter_enabled=False):
        self.filter_enabled = filter_enabled
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.block_size_input = self.input_rate // self.blocks_per_second
        self.pa = pyaudio.PyAudio()
        self.stream = self._create_stream(file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def _create_stream(self, file):
        def callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            self.buffer_queue.put(in_data)
            return (None, pyaudio.paContinue)

        kwargs = {
            'format': self.format,
            'channels': self.channels,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': callback
        }

        self.chunk = None
        if self.device:  # non-default device selected
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        stream = self.pa.open(**kwargs)
        stream.start_stream()

        return stream

    def _end_stream(self, stream):
        stream.stop_stream()
        stream.close()

    def _destroy(self):
        self._end_stream(self.stream)
        self.pa.terminate()

    def _resample(self, data):
        """
        The user's microphone may not support the native processing sampling rate, so resample from input_rate to sample_rate here for webrtcvad and deepspeech.
        """
        data16 = np.frombuffer(data, dtype=np.int16)
        resample_size = int(len(data16) * self.sample_rate / self.input_rate)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tobytes()

    filter_enabled = False
    lowpass_frequency = 75  # 75  # 100  # 75  # 50
    highpass_frequency = 6000  # 6000  # 7000  # 7999

    def _filter(self, data):
        data16 = np.frombuffer(data, dtype=np.int16)

        nyquist_frequency = 0.5 * self.sample_rate
        b, a = signal.filter_design.butter(1, [
            self.lowpass_frequency / nyquist_frequency,
            self.highpass_frequency / nyquist_frequency
        ], btype='bandpass')

        filtered, _ = signal.signaltools.lfilter(
            b, a, data16, axis=0, zi=signal.signaltools.lfilter_zi(b, a))

        return np.array(filtered, dtype=np.int16).tobytes()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate)

    def _write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)  # wf.setsampwidth(self.pa.get_sample_size(format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

    def _frames(self):
        """Generator that yields all audio frames from microphone, blocking if necessary."""
        if self.input_rate == self.sample_rate:
            if self.filter_enabled:
                while True:
                    yield self._filter(self.buffer_queue.get())
            else:
                while True:
                    yield self.buffer_queue.get()
        else:
            while True:
                if self.filter_enabled:
                    yield self._filter(self._resample(self.buffer_queue.get()))
                else:
                    yield self._resample(self.buffer_queue.get())

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
            for frame in self._frames():
                if len(frame) < 640:
                    return

                is_speech = self.vad.is_speech(frame, self.sample_rate)

                if triggered:
                    yield frame
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len(
                        [f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > ratio * ring_buffer.maxlen:
                        triggered = False
                        yield None
                        ring_buffer.clear()
                else:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len(
                        [f for f, speech in ring_buffer if speech])
                    if num_voiced > ratio * ring_buffer.maxlen:
                        triggered = True
                        for f, s in ring_buffer:
                            yield f
                        ring_buffer.clear()

        except KeyboardInterrupt:
            pass
        finally:
            self._destroy()


class Transcriber:

    def __init__(self, args):
        # Load the DeepSpeech model
        self.model = self.init_deepspeech(args.model, args.scorer)

        # Start audio
        self.vas = VoiceAudioService(aggressiveness=args.aggressiveness,
                                     device=args.device,
                                     input_rate=args.rate,
                                     file=args.file,
                                     filter_enabled=args.filter)

    def init_deepspeech(self, model_path, scorer_path):
        print("Initialising model: %s" % model_path)
        logging.info("Initialising model: %s", scorer_path)

        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'output_graph.pb')
            scorer_path = os.path.join(model_path, scorer_path)

        model = deepspeech.Model(model_path)
        if scorer_path:
            logging.info("Scorer: %s", scorer_path)
            model.enableExternalScorer(scorer_path)

        return model

    def transcribe(self, nospinner, savewav):
        if savewav:
            os.makedirs(savewav, exist_ok=True)

        spinner = None
        if not nospinner:
            spinner = halo.Halo(spinner='line')

        stream_context = self.model.createStream()
        wav_data = bytearray()
        for utterance in self.vas.utterances():
            if utterance is not None:
                if spinner:
                    spinner.start()

                logging.debug("start utterance")
                stream_context.feedAudioContent(
                    np.frombuffer(utterance, np.int16))
                if savewav:
                    wav_data.extend(utterance)
            else:
                if spinner:
                    spinner.stop()

                logging.debug("end utterence")

                text = stream_context.finishStream()
                if text:
                    if savewav:
                        self.vas._write_wav(os.path.join(savewav, datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S - " + text + ".wav")), wav_data)

                    yield text

                if savewav:
                    wav_data = bytearray()

                stream_context = self.model.createStream()


def main(args):
    transcriber = Transcriber(args)

    print("Listening (Ctrl-C to exit).")
    # Stream from microphone to DeepSpeech using VAD
    for text in transcriber.transcribe(args.nospinner, args.savewav):
        if text:
            print("Recognised: %s" % text)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-a', '--aggressiveness', type=int, default=2,
                        help="Set the voice activity detection aggressiveness: an integer between 0 and 3, with 0 being the least aggressive at filtering out non-speech and 3 the most aggressive. Default: 2.")

    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner.")

    parser.add_argument('-F', '--filter', action='store_true',
                        help="Enable the bandpass filter.")

    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to a given directory.")

    parser.add_argument('-f', '--file',
                        help="Read from a .wav file instead of the microphone.")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model).")

    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")

    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")

    parser.add_argument('-r', '--rate', type=int, default=VoiceAudioService.sample_rate,
                        help=f"Input device sample rate. Default: {VoiceAudioService.sample_rate}. Your device may require 44100.")

    main(parser.parse_args())
