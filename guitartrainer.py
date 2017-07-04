# MIT License
#
# Copyright (c) 2017 VinnieM
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pyaudio
import numpy as np
import Tkinter
from Tkinter import Listbox, Label, Radiobutton, Spinbox, DoubleVar, IntVar, Scrollbar
import tkFont
import random
from threading import Thread
import sched
import time
from collections import deque

"""The sampling frequency is the typical standard 44100.  The sample size was chosen to
to be able to reliably distinguish the closely spaced (in terms of frequency) notes on the
low E string, yet still be processed quick enough so that about three audio samples could be
processed per second with modest computer hardware"""
SAMPLING_FREQ = 44100
SAMPLE_SIZE = 2 ** 14

"""These midi min and max values are used to set the fft bin range to match the frequency range
of a 6-string guitar that is using standard tuning.  Frequencies above and below the fft bins are ignored.
This acts like a bandpass filter."""
MIDI_MIN = 40  # E2
MIDI_MAX = 76  # E5 - decided to ignore frets above 12 on the high E because having trouble capturing.  Will revisit

"""This value is used to adjust the noise threshold value.  A value of 1.3 means a note will not be
captured until it is 30% above the sensed noise floor.  The application constantly listens and adjust
the noise floor value.  The 1.3 value was found to be best for a simple/cheap microphone placed inside the guitar's
sound hole.  If you're using a proper pickup/amp you may be able to increase this value and avoid the application
mistaking noise for a note."""
THRESHOLD_MULTIPLIER_DEFAULT = 1.3

ADD_NOTES_INCREMENT_DEFAULT = 10


class Note(object):
    """
    This class holds a single note which has a midi value, a name (e.g. A#, F, Gb, etc.), and a staff
    location (i.e. number of 0.5 staff lines relative to guitar middle-C).  The class uses a scale with A = 440Hz.
    """

    __slots__ = ['_midi', '_name', '_staff_loc']

    def __init__(self, midi, name, staff_loc):
        if midi >= MIDI_MIN or midi <= MIDI_MAX:
            self._midi = midi
            self._name = name
            self._staff_loc = staff_loc
        else:
            raise ValueError

    @property
    def midi(self):
        return self._midi

    @property
    def name(self):
        return self._name

    @property
    def staff_loc(self):
        """ Distance (in 1/2 staff lines) from "guitar middle-C"
        (i.e. midi 48, one octave lower than piano middle-C)
        """
        return self._staff_loc

    @property
    def freq(self):
        return Note.midi_to_freq(self._midi)

    @staticmethod
    def midi_to_freq(midi):
        """ function converts a midi value to a frequency """
        return 440 * 2.0 ** ((midi - 69) / 12.0)

    @staticmethod
    def freq_to_midi(freq):
        """ function converts a frequency to a midi value """
        return int(round(69 + 12 * np.log2(freq / 440.0)))

    def __str__(self):
        return self.name + ", " + str(self._midi)

    def __repr__(self):
        return self.name + ", " + str(self._midi)

    def __eq__(self, other):
        return self._midi == other.midi and self._name == other.name and self._staff_loc == other.staff_loc


class Topic(object):
    """ This class represent a 'learning unit'.  For instance
    B String Notes could be a topic.  The topic name is 'B String Notes', and
    it has multiple Notes.  Currently, the Topic class doesn't do much and could easily
    be replaced with a simple list/dictionary, but we may add some functionality in the
    future, so we will leave it as-is for now.
    """
    __slots__ = ['_name', '_notes']

    def __init__(self, name, notes):
        self._name = name
        self._notes = notes

    @property
    def name(self):
        return self._name

    @property
    def notes(self):
        return list(self._notes)


class StringTopic(Topic):
    """ In the future we may have extra functionality in this child class since certain types of topics
    may have to be handled differently.  Maybe we should add the string/fret location to each StringNote
    for display purposes?"""

    def __init__(self, name, notes):
        super(StringTopic, self).__init__(name, notes)


class Trainer(object):
    """  Trainer contains a list of Topics and keeps track of practice notes as they are run"""

    __slots__ = ['_que', '_noise_threshold', '_freq_step', '_fft_freqs', '_fft_imin', '_fft_imax',
                 '_audio_buf', '_window_func', '_audio_stream', '_test_scale', '_noise_levels',
                 '_topics', '_topic', '_topic_changed', '_notes_in_play', '_notes_in_queue',
                 '_practiced_notes', '_add_notes_method', '_add_notes_method_changed',
                 '_threshold_multiplier', '_add_notes_increment']

    ADD_NOTES_INCREMENTALLY = 0
    ADD_NOTES_ALL_AT_ONCE = 1

    def __init__(self):
        #  print "Entering Trainer: __init__"
        self._noise_threshold = 100000  # initial setting
        self._noise_levels = deque([self._noise_threshold] * 50)
        self._topics = [StringTopic('Low E String Sans Sharps/Flats',
                                    [Note(40, 'E', -5), Note(41, 'F', -4), Note(43, 'G', -3), Note(45, 'A', -2),
                                     Note(47, 'B', -1), Note(48, 'C', 0), Note(50, 'D', 1), Note(52, 'E', 2)]),
                        StringTopic('Low E String Incl Sharps',
                                    [Note(40, 'E', -5), Note(41, 'F', -4), Note(42, 'F#', -4), Note(43, 'G', -3),
                                     Note(44, 'G#', -3), Note(45, 'A', -2), Note(46, 'A#', -2), Note(47, 'B', -1),
                                     Note(48, 'C', 0), Note(49, 'C#', 0), Note(50, 'D', 1), Note(51, 'D#', 1),
                                     Note(52, 'E', 2)]),
                        StringTopic('Low E String Incl Flats',
                                    [Note(40, 'E', -5), Note(41, 'F', -4), Note(42, 'Gb', -3), Note(43, 'G', -3),
                                     Note(44, 'Ab', -2), Note(45, 'A', -2), Note(46, 'Bb', -1), Note(47, 'B', -1),
                                     Note(48, 'C', 0), Note(49, 'Db', 1), Note(50, 'D', 1), Note(51, 'Eb', 2),
                                     Note(52, 'E', 2)]),
                        StringTopic('A String Sans Sharps/Flats',
                                    [Note(45, 'A', -2), Note(47, 'B', -1), Note(48, 'C', 0), Note(50, 'D', 1),
                                     Note(52, 'E', 2), Note(53, 'F', 3), Note(55, 'G', 4), Note(57, 'A', 5)]),
                        StringTopic('A String Incl Sharps',
                                    [Note(45, 'A', -2), Note(46, 'A#', -2), Note(47, 'B', -1), Note(48, 'C', 0),
                                     Note(49, 'C#', 0), Note(50, 'D', 1), Note(51, 'D#', 1), Note(52, 'E', 2),
                                     Note(53, 'F', 3), Note(54, 'F#', 3), Note(55, 'G', 4), Note(56, 'G#', 4),
                                     Note(57, 'A', 5)]),
                        StringTopic('A String Incl Flats',
                                    [Note(45, 'A', -2), Note(46, 'Bb', -1), Note(47, 'B', -1), Note(48, 'C', 0),
                                     Note(49, 'Db', 1), Note(50, 'D', 1), Note(51, 'Eb', 2), Note(52, 'E', 2),
                                     Note(53, 'F', 3), Note(54, 'Gb', 4), Note(55, 'G', 4), Note(56, 'Ab', 5),
                                     Note(57, 'A', 5)]),
                        StringTopic('D String Sans Sharps/Flats',
                                    [Note(50, 'D', 1), Note(52, 'E', 2), Note(53, 'F', 3), Note(55, 'G', 4),
                                     Note(57, 'A', 5), Note(59, 'B', 6), Note(60, 'C', 7), Note(62, 'D', 8)]),
                        StringTopic('D String Incl Sharps',
                                    [Note(50, 'D', 1), Note(51, 'D#', 1), Note(52, 'E', 2), Note(53, 'F', 3),
                                     Note(54, 'F#', 3), Note(55, 'G', 4), Note(56, 'G#', 4), Note(57, 'A', 5),
                                     Note(58, 'A#', 5), Note(59, 'B', 6), Note(60, 'C', 7), Note(61, 'C#', 7),
                                     Note(62, 'D', 8)]),
                        StringTopic('D String Incl Flats',
                                    [Note(50, 'D', 1), Note(51, 'Eb', 2), Note(52, 'E', 2), Note(53, 'F', 3),
                                     Note(54, 'Gb', 4), Note(55, 'G', 4), Note(56, 'Ab', 5), Note(57, 'A', 5),
                                     Note(58, 'Bb', 6), Note(59, 'B', 6), Note(60, 'C', 7), Note(61, 'Db', 8),
                                     Note(62, 'D', 8)]),
                        StringTopic('G String Sans Sharps/Flats',
                                    [Note(55, 'G', 4), Note(57, 'A', 5), Note(59, 'B', 6), Note(60, 'C', 7),
                                     Note(62, 'D', 8), Note(64, 'E', 9), Note(65, 'F', 10), Note(67, 'G', 11)]),
                        StringTopic('G String Incl Sharps',
                                    [Note(55, 'G', 4), Note(56, 'G#', 4), Note(57, 'A', 5), Note(58, 'A#', 5),
                                     Note(59, 'B', 6), Note(60, 'C', 7), Note(61, 'C#', 7), Note(62, 'D', 8),
                                     Note(63, 'D#', 8), Note(64, 'E', 9), Note(65, 'F', 10), Note(66, 'F#', 10),
                                     Note(67, 'G', 11)]),
                        StringTopic('G String Incl Flats',
                                    [Note(55, 'G', 4), Note(56, 'Ab', 5), Note(57, 'A', 5), Note(58, 'Bb', 6),
                                     Note(59, 'B', 6), Note(60, 'C', 7), Note(61, 'Db', 8), Note(62, 'D', 8),
                                     Note(63, 'Eb', 9), Note(64, 'E', 9), Note(65, 'F', 10), Note(66, 'Gb', 11),
                                     Note(67, 'G', 11)]),
                        StringTopic('B String Sans Sharps/Flats',
                                    [Note(59, 'B', 6), Note(60, 'C', 7), Note(62, 'D', 8), Note(64, 'E', 9),
                                     Note(65, 'F', 10), Note(67, 'G', 11), Note(69, 'A', 12), Note(71, 'B', 13)]),
                        StringTopic('B String Incl Sharps',
                                    [Note(59, 'B', 6), Note(60, 'C', 7), Note(61, 'C#', 7), Note(62, 'D', 8),
                                     Note(63, 'D#', 8), Note(64, 'E', 9), Note(65, 'F', 10), Note(66, 'F#', 10),
                                     Note(67, 'G', 11), Note(68, 'G#', 11), Note(69, 'A', 12), Note(70, 'A#', 12),
                                     Note(71, 'B', 13)]),
                        StringTopic('B String Incl Flats',
                                    [Note(59, 'B', 6), Note(60, 'C', 7), Note(61, 'Db', 8), Note(62, 'D', 8),
                                     Note(63, 'Eb', 9), Note(64, 'E', 9), Note(65, 'F', 10), Note(66, 'Gb', 11),
                                     Note(67, 'G', 11), Note(68, 'Ab', 12), Note(69, 'A', 12), Note(70, 'Bb', 13),
                                     Note(71, 'B', 13)]),
                        StringTopic('High E String Sans Sharps/Flats',
                                    [Note(64, 'E', 9), Note(65, 'F', 10), Note(67, 'G', 11), Note(69, 'A', 12),
                                     Note(71, 'B', 13), Note(72, 'C', 14), Note(74, 'D', 15), Note(76, 'E', 16)]),
                        StringTopic('High E String Incl Sharps',
                                    [Note(64, 'E', 9), Note(65, 'F', 10), Note(66, 'F#', 10), Note(67, 'G', 11),
                                     Note(68, 'G#', 11), Note(69, 'A', 12), Note(70, 'A#', 12), Note(71, 'B', 13),
                                     Note(72, 'C', 14), Note(73, 'C#', 14), Note(74, 'D', 15), Note(75, 'D#', 15),
                                     Note(76, 'E', 16)]),
                        StringTopic('High E String Incl Flats',
                                    [Note(64, 'E', 9), Note(65, 'F', 10), Note(66, 'Gb', 11), Note(67, 'G', 11),
                                     Note(68, 'Ab', 12), Note(69, 'A', 12), Note(70, 'Bb', 13), Note(71, 'B', 13),
                                     Note(72, 'C', 14), Note(73, 'Db', 15), Note(74, 'D', 15), Note(75, 'Eb', 16),
                                     Note(76, 'E', 16)])]
        self._topic = self._topics[1]
        self._add_notes_method = Trainer.ADD_NOTES_INCREMENTALLY
        self._notes_in_play = []
        self._notes_in_queue = []
        self._practiced_notes = []
        self._topic_changed = True
        self._add_notes_method_changed = False
        self._threshold_multiplier = THRESHOLD_MULTIPLIER_DEFAULT
        self._add_notes_increment = ADD_NOTES_INCREMENT_DEFAULT
        self._freq_step = float(SAMPLING_FREQ) / SAMPLE_SIZE
        self._fft_freqs = np.fft.fftfreq(SAMPLE_SIZE, 1. / SAMPLING_FREQ)
        self._fft_imin = (np.abs(self._fft_freqs - Note.midi_to_freq(MIDI_MIN))).argmin()
        self._fft_imax = (np.abs(self._fft_freqs - Note.midi_to_freq(MIDI_MAX))).argmin() + 1
        self._audio_buf = np.zeros(SAMPLE_SIZE, dtype=np.float32)
        self._window_func = np.hanning(SAMPLE_SIZE)
        self._audio_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                                    channels=1,
                                                    rate=SAMPLING_FREQ,
                                                    input=True,
                                                    frames_per_buffer=SAMPLE_SIZE)
        #  print "Exiting Trainer: __init__"

    def new_note_practice(self):
        """  This function picks the next note to practice.
        """
        # check if topic or add_notes_method has changed.
        if self._topic_changed or self._add_notes_method_changed:
            self._topic_changed = False
            self._add_notes_method_changed = False
            self._practiced_notes = []
            if self._add_notes_method == Trainer.ADD_NOTES_INCREMENTALLY:  # slowly add new notes to practice
                self._notes_in_queue = self._topic.notes
                self._notes_in_play = []
                for x in range(0, 3):  # lets pick the first three notes to practice.
                    r = random.choice(self._notes_in_queue)
                    self._notes_in_play.append(r)
                    del (self._notes_in_queue[self._notes_in_queue.index(r)])
            elif self._add_notes_method == Trainer.ADD_NOTES_ALL_AT_ONCE:  # practice all the notes in the topic
                self._notes_in_queue = []
                self._notes_in_play = self._topic.notes
        elif (len(self._practiced_notes) > 0 and len(self._practiced_notes) % self._add_notes_increment == 0
              and len(self._notes_in_queue) > 0):  # we are adding notes incrementally and have notes still in queue
            r = random.choice(self._notes_in_queue)
            self._notes_in_play.append(r)
            del (self._notes_in_queue[self._notes_in_queue.index(r)])

        # Let's pick a note to play
        if len(self._practiced_notes) >= 4:
            notes = list(self._notes_sorted_by_times_practiced())  # pick one of the least practiced notes
            notes.remove(self.current_note_practice.target_note)  # but don't repeat the last note
            r = notes[random.randint(0, min((len(notes) - 1), 2))]  # pick from the 3 least played notes
        elif len(self._practiced_notes) > 0:  # don't repeat the last note
            notes = list(self._notes_in_play)
            notes.remove(self.current_note_practice.target_note)
            r = random.choice(notes)
        else:  # it is the first note for this topic (or add notes method has changed) so pick any note
            r = random.choice(self._notes_in_play)

        # Create a new practice note and start the audio capture
        note_practice = NotePractice(r)
        self._practiced_notes.append(note_practice)
        t1 = Thread(target=self._capture_note_thread, args=(note_practice,))
        t1.start()

    def _notes_sorted_by_times_practiced(self):
        """sort the practiced notes by the number of times they were chosen.  This
        ensures we practice each note about the same number of times"""
        note_times = [{'Note': n, 'Times': 0} for n in self._notes_in_play]
        counter = 0
        for note_practice in reversed(self._practiced_notes):
            counter += 1
            if counter > len(note_times) * 5:  # don't look too far back in time. 5x the number of notes is good enough
                break
            for i in note_times:
                if i['Note'] == note_practice.target_note:
                    i['Times'] += 1
                    break
        note_times.sort(key=lambda x: x['Times'])
        return [n['Note'] for n in note_times]

    def _notes_sorted_by_elapsed_time(self):
        """might use this function to adjust notes practiced by how QUICKLY the student is playing the correct note"""
        note_times = [{'Note': n, 'Times': 0, 'Avg_Elapsed_Time': 0} for n in self._notes_in_play]
        counter = 0
        for note_practice in reversed(self._practiced_notes):  # start with the most recent practice note
            counter += 1
            if counter > len(note_times) * 5:  # don't look too far back in time. 5x the number of notes is good enough
                break
            for i in note_times:
                if i['Note'] == note_practice.target_note:
                    i['Times'] += 1
                    i['Avg_Elapsed_Time'] = float((i['Avg_Elapsed_Time'] * (i['Times'] - 1))
                                                  + min(5.0, note_practice.elapsed_time)) / i[
                                                'Times']  # set max to 5secs
                    break
        note_times.sort(key=lambda x: x['Avg_Elapsed_Time'], reverse=True)
        return [n['Note'] for n in note_times]

    @property
    def noise_threshold(self):
        return self._noise_threshold

    @property
    def threshold_multiplier(self):
        return self._threshold_multiplier

    @threshold_multiplier.setter
    def threshold_multiplier(self, value):
        if 1 <= value <= 3:
            self._threshold_multiplier = value
        else:
            raise ValueError

    @property
    def add_notes_increment(self):
        return self._add_notes_increment

    @add_notes_increment.setter
    def add_notes_increment(self, value):
        if 1 <= value <= 100:
            self._add_notes_increment = value
        else:
            raise ValueError

    @property
    def topics(self):
        return list(self._topics)

    @property
    def current_topic(self):
        return self._topic

    @property
    def notes_in_play(self):
        return list(self._notes_in_play)

    @property
    def notes_in_queue(self):
        return list(self._notes_in_queue)

    @property
    def add_notes_method(self):
        return self._add_notes_method

    @add_notes_method.setter
    def add_notes_method(self, value):
        """The application will either train the student on all the notes of the Topic at once
        or will start with three notes and slowly increase the number of notes over time."""
        if value == Trainer.ADD_NOTES_INCREMENTALLY or value == Trainer.ADD_NOTES_ALL_AT_ONCE:
            if value != self._add_notes_method:
                self._add_notes_method_changed = True
            self._add_notes_method = value
        else:
            raise ValueError

    @current_topic.setter
    def current_topic(self, value):
        if value <= len(self.topics):
            self._topic = self._topics[value]
            self._topic_changed = True
        else:
            raise ValueError

    @property
    def current_note_practice(self):
        if len(self._practiced_notes) > 0:
            return self._practiced_notes[-1]
        else:
            return None

    @property
    def current_target_note(self):
        if len(self._practiced_notes) > 0:
            return self._practiced_notes[-1].target_note
        else:
            return None

    def kill_current_note_practice(self):
        if self.current_note_practice is not None:
            self.current_note_practice.terminate = True

    def _capture_note_thread(self, note_practice):
        """This thread captures a note heard"""
        #  print "Entering _capture_note_thread"
        self._audio_stream.start_stream()
        note_practice.start_timestamp = time.time()
        while self._audio_stream.is_active() and note_practice.terminate is False:
            self._audio_buf = np.fromstring(self._audio_stream.read(SAMPLE_SIZE), np.int16)
            fft = np.fft.rfft(self._audio_buf * self._window_func)
            freq = self._fft_freqs[np.abs(fft[self._fft_imin:self._fft_imax]).argmax() + self._fft_imin]
            peak = np.average(np.abs(fft))
            self._noise_levels.popleft()
            self._noise_levels.append(peak)
            self._noise_threshold = np.average(sorted(self._noise_levels)[0:10]) * self._threshold_multiplier
            if peak > self._noise_threshold:
                note_practice.num_notes_heard += 1
                if Note.freq_to_midi(freq) == note_practice.target_note.midi:
                    note_practice.success_timestamp = time.time()
                    note_practice.complete = True
                    self._audio_stream.stop_stream()
                    break
        #  print "Exiting _capture_note_thread"


class NotePractice(object):
    """Each note requested is considered an practice note.  A NotePractice has a Note and records how many notes
    were heard before the correct note was played and records how long it took to play the correct note."""

    __slots__ = ['_target_note', '_num_notes_heard', '_complete',
                 '_terminate', '_start_timestamp', '_success_timestamp']

    def __init__(self, target_note):
        # print "Entering NotePractice: __init__"
        self._target_note = target_note
        self._complete = False
        self._num_notes_heard = 0
        self._start_timestamp = 0
        self._success_timestamp = 0
        self._terminate = False
        # print "Exiting NotePractice: __init__"

    @property
    def target_note(self):
        return self._target_note

    @property
    def terminate(self):
        return self._terminate

    @terminate.setter
    def terminate(self, value):
        if self._terminate is True or value is False:  # should not be asking twice
            raise ValueError
        self._terminate = value

    @property
    def complete(self):
        return self._complete

    @complete.setter
    def complete(self, value):
        if self._complete is False and value is True:
            self._complete = value
        else:
            raise ValueError

    @property
    def num_notes_heard(self):
        return self._num_notes_heard

    @num_notes_heard.setter
    def num_notes_heard(self, value):
        self._num_notes_heard = value

    @property
    def start_timestamp(self):
        return self._start_timestamp

    @start_timestamp.setter
    def start_timestamp(self, value):
        if self._start_timestamp != 0:  # we shouldn't be setting the start_timestamp more than once.
            raise ValueError
        self._start_timestamp = value

    @property
    def success_timestamp(self):
        return self._success_timestamp

    @success_timestamp.setter
    def success_timestamp(self, value):
        if value <= self._start_timestamp or self._success_timestamp != 0:  # must come after the start and be set once.
            raise ValueError
        self._success_timestamp = value

    @property
    def elapsed_time(self):
        if self.complete:
            return self._success_timestamp - self._start_timestamp
        else:
            return time.time() - self._start_timestamp  # still waiting for target note to be heard


class MainGUI(object):
    def __init__(self):
        # print "Entering MainGUI: __init__"
        self._trainer = Trainer()
        self._in_training = False
        self._target_note_oval = None
        self._sharp_flat_symbol = None
        self._scheduler = sched.scheduler(time.time, time.sleep)

        self._gui_top = Tkinter.Tk()
        self._gui_top.title("Guitar Fretboard Trainer")
        self._gui_top.resizable(width=False, height=False)

        self._training_topic_list_label = Label(self._gui_top, text="<- TOPICS ->", font="Verdana 10 bold")
        self._training_topic_list = Listbox(self._gui_top, width=35, selectmode="single", font="Verdana 10 bold")
        self._training_topic_scroll = Scrollbar(self._gui_top)
        self._training_topic_list.config(yscrollcommand=self._training_topic_scroll.set)
        self._training_topic_scroll.config(command=self._training_topic_list.yview)

        i = 0
        for topic in self._trainer.topics:
            self._training_topic_list.insert(i, topic.name)
            i += 1
        self._training_topic_list.configure(exportselection=False)
        self._training_topic_list.select_set(1)
        self._training_topic_list.bind('<<ListboxSelect>>', self._on_set_topic)

        self._canvas_label = Label(self._gui_top, text="<- STAFF ->", font="Verdana 10 bold")
        self._canvas = Tkinter.Canvas(self._gui_top, bg="white", height=150, width=200)

        self._high_g_ledger = self._canvas.create_line((85, 20, 115, 20), fill="black")
        self._high_e_ledger = self._canvas.create_line((85, 30, 115, 30), fill="black")
        self._high_c_ledger = self._canvas.create_line((85, 40, 115, 40), fill="black")
        self._high_a_ledger = self._canvas.create_line((85, 50, 115, 50), fill="black")

        self._canvas.create_line((40, 60, 160, 60), fill="black")
        self._canvas.create_line((40, 70, 160, 70), fill="black")
        self._canvas.create_line((40, 80, 160, 80), fill="black")
        self._canvas.create_line((40, 90, 160, 90), fill="black")
        self._canvas.create_line((40, 100, 160, 100), fill="black")

        self._middle_c_ledger = self._canvas.create_line((85, 110, 115, 110), fill="black")
        self._low_a_ledger = self._canvas.create_line((85, 120, 115, 120), fill="black")
        self._low_f_ledger = self._canvas.create_line((85, 130, 115, 130), fill="black")

        self._status_label = Label(self._gui_top, text="<- STATUS ->", font="Verdana 10 bold")
        self._status = Tkinter.Canvas(self._gui_top, bg="black", height=150, width=300)
        self._status_txt1 = self._status.create_text(10, 10, anchor=Tkinter.NW, text="",
                                                     font="Verdana 10 bold", fill="green")
        self._status_txt2 = self._status.create_text(10, 30, anchor=Tkinter.NW, text="",
                                                     font="Verdana 10 bold", fill="green")
        self._status_txt3 = self._status.create_text(10, 50, anchor=Tkinter.NW, text="",
                                                     font="Verdana 10 bold", fill="green")

        # add note method
        self._add_note_label = Label(self._gui_top, text="Add Note Method:", font="Verdana 10 bold")
        self._add_note_method_inc_radio = Radiobutton(self._gui_top, text="Incrementally",
                                                      font="Verdana 10 bold",
                                                      command=self._on_note_method_change_inc, value=1)
        self._add_note_method_all_radio = Radiobutton(self._gui_top, text="All-at-Once",
                                                      font="Verdana 10 bold",
                                                      command=self._on_note_method_change_all, value=2)

        if self._trainer.add_notes_method == self._trainer.ADD_NOTES_INCREMENTALLY:
            self._add_note_method_inc_radio.select()
        elif self._trainer.add_notes_method == self._trainer.ADD_NOTES_ALL_AT_ONCE:
            self._add_note_method_all_radio.select()
        else:
            raise ValueError

        # add threshold setter
        self._threshold_multiplier = DoubleVar(self._gui_top)
        self._threshold_mult_label = Label(self._gui_top, text="Noise\nThreshold:", font="Verdana 10 bold")
        self._threshold_mult_spin = Spinbox(self._gui_top, width=5, font="Verdana 10 bold",
                                            values=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                    1.6, 1.7, 1.8, 1.9, 2.0,
                                                    2.1, 2.2, 2.3, 2.4, 2.5),
                                            textvariable=str(self._threshold_multiplier),
                                            command=self._on_threshold_change)
        self._threshold_multiplier.set(self._trainer.threshold_multiplier)

        # add note increment setter
        self._add_notes_increment = IntVar(self._gui_top)
        self._add_notes_increment_label = Label(self._gui_top, text="Note\nIncrement:", font="Verdana 10 bold")
        self._add_notes_increment_spin = Spinbox(self._gui_top, width=5, font="Verdana 10 bold",
                                                 values=(10, 20, 30, 40, 50),
                                                 textvariable=str(self._add_notes_increment),
                                                 command=self._on_add_notes_increment_change)
        self._add_notes_increment.set(self._trainer.add_notes_increment)

        # add buttons
        self._start_button = Tkinter.Button(self._gui_top, text="Start", font="Verdana 10 bold",
                                            command=self._start_training)
        self._exit_button = Tkinter.Button(self._gui_top, text="Exit", font="Verdana 10 bold",
                                           command=self._on_closing)

        # layout the widgets
        self._training_topic_list_label.grid(row=0, column=1,
                                             columnspan=1, rowspan=1,
                                             sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                                             padx=5, pady=2)

        self._training_topic_list.grid(row=1, column=1,
                                       columnspan=1, rowspan=1,
                                       sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                                       padx=5, pady=2)

        self._training_topic_scroll.grid(row=1, column=1,
                                         columnspan=1, rowspan=1,
                                         sticky=Tkinter.E + Tkinter.N + Tkinter.S)

        self._canvas_label.grid(row=0, column=2,
                                columnspan=1, rowspan=1,
                                sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                                padx=5, pady=2)
        self._canvas.grid(row=1, column=2,
                          columnspan=1, rowspan=1,
                          sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                          padx=5, pady=2)

        self._status_label.grid(row=0, column=3,
                                columnspan=1, rowspan=1,
                                sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                                padx=5, pady=2)
        self._status.grid(row=1, column=3,
                          columnspan=1, rowspan=1,
                          sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                          padx=5, pady=2)

        self._add_note_label.grid(row=2, column=1,
                                  columnspan=1, rowspan=1,
                                  sticky=Tkinter.W + Tkinter.N + Tkinter.S,
                                  padx=5, pady=5)
        self._add_note_method_inc_radio.grid(row=3, column=1,
                                             columnspan=1, rowspan=1,
                                             sticky=Tkinter.W + Tkinter.N + Tkinter.S,
                                             padx=5, pady=0)
        self._add_note_method_all_radio.grid(row=4, column=1,
                                             columnspan=1, rowspan=1,
                                             sticky=Tkinter.W + Tkinter.N + Tkinter.S,
                                             padx=5, pady=0)

        self._threshold_mult_label.grid(row=2, column=2,
                                        columnspan=1, rowspan=1,
                                        sticky=Tkinter.W + Tkinter.N + Tkinter.S,
                                        padx=5, pady=5)
        self._threshold_mult_spin.grid(row=3, column=2,
                                       columnspan=1, rowspan=2,
                                       sticky=Tkinter.W + Tkinter.N + Tkinter.S,
                                       padx=15, pady=10)

        self._add_notes_increment_label.grid(row=2, column=2,
                                             columnspan=1, rowspan=1,
                                             sticky=Tkinter.E + Tkinter.N + Tkinter.S,
                                             padx=5, pady=5)
        self._add_notes_increment_spin.grid(row=3, column=2,
                                            columnspan=1, rowspan=2,
                                            sticky=Tkinter.E + Tkinter.N + Tkinter.S,
                                            padx=14, pady=10)

        self._start_button.grid(row=2, column=3,
                                columnspan=1, rowspan=2,
                                sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                                padx=85, pady=15)
        self._exit_button.grid(row=4, column=3,
                               columnspan=1, rowspan=2,
                               sticky=Tkinter.W + Tkinter.E + Tkinter.N + Tkinter.S,
                               padx=85, pady=5)

        self._gui_top.protocol("WM_DELETE_WINDOW", self._on_closing)

        #  print "Exiting MainGUI: __init__"
        self._gui_top.mainloop()

    def _start_training(self):
        #  print "Entering _start_training"
        if not self._in_training and (self._trainer.current_note_practice is None
                                      or self._trainer.current_note_practice.complete):
            self._trainer.new_note_practice()
            self._canvas.delete(self._high_g_ledger)
            self._canvas.delete(self._high_e_ledger)
            self._canvas.delete(self._high_c_ledger)
            self._canvas.delete(self._high_a_ledger)
            self._canvas.delete(self._middle_c_ledger)
            self._canvas.delete(self._low_a_ledger)
            self._canvas.delete(self._low_f_ledger)
            self._canvas.delete(self._target_note_oval)
            self._canvas.delete(self._sharp_flat_symbol)
            # self._status.delete(self._status_txt1)  # leave last note in status window?
            self._status.delete(self._status_txt2)
            self._status.delete(self._status_txt3)

            staff_loc = self._trainer.current_target_note.staff_loc

            # draw ledger lines if needed, depending on the note to be played
            if staff_loc >= 12:
                self._high_a_ledger = self._canvas.create_line((85, 50, 115, 50), fill="black")
            if staff_loc >= 14:
                self._high_c_ledger = self._canvas.create_line((85, 40, 115, 40), fill="black")
            if staff_loc >= 16:
                self._high_e_ledger = self._canvas.create_line((85, 30, 115, 30), fill="black")
            if staff_loc >= 18:
                self._high_g_ledger = self._canvas.create_line((85, 20, 115, 20), fill="black")

            if staff_loc <= 0:
                self._middle_c_ledger = self._canvas.create_line((85, 110, 115, 110), fill="black")
            if staff_loc <= -2:
                self._low_a_ledger = self._canvas.create_line((85, 120, 115, 120), fill="black")
            if staff_loc <= -4:
                self._low_f_ledger = self._canvas.create_line((85, 130, 115, 130), fill="black")

            self._target_note_oval = self._canvas.create_oval(93, 106 - (5 * staff_loc),
                                                              107, 114 - (5 * staff_loc), width=3)

            # add a '#' or 'b' symbol to the note if necessary
            if len(self._trainer.current_target_note.name) > 1:
                symbol = self._trainer.current_target_note.name[1:]
                self._sharp_flat_symbol = self._canvas.create_text(90, 118 - (5 * staff_loc),
                                                                   font=tkFont.Font(family='Helvetica',
                                                                                    size=16),
                                                                   anchor=Tkinter.SE, text=symbol)

            txt2 = "Notes in Play:\n" + self._note_names_on()
            self._status_txt2 = self._status.create_text(10, 40, anchor=Tkinter.NW, text=txt2,
                                                         font="Verdana 10 bold", fill="green")
            txt3 = "Notes in Queue:\n" + self._note_names_off()
            self._status_txt3 = self._status.create_text(10, 80, anchor=Tkinter.NW, text=txt3,
                                                         font="Verdana 10 bold", fill="green")

            self._in_training = True
            self._check_note_practice_status()
        # print "Exiting _start_training"

    def _check_note_practice_status(self):
        """check if the correct note was played which means the note practice is complete and it's time for another
        practice note.  Display the note in green so that the user sees the correct note was played, wait one second
        and ask the trainer for another note."""
        #  print "Entering _check_note_practice_status"
        if self._in_training:
            if self._trainer.current_note_practice.complete:
                self._status.delete(self._status_txt1)
                self._status.delete(self._status_txt2)
                self._status.delete(self._status_txt3)
                txt1 = self._trainer.current_target_note.name + ", "
                txt1 += str(round(self._trainer.current_note_practice.elapsed_time, 2))
                txt1 += "secs"
                self._status_txt1 = self._status.create_text(10, 10, anchor=Tkinter.NW, text=txt1,
                                                             font="Verdana 10 bold", fill="green")
                txt2 = "Notes in Play:\n" + self._note_names_on()
                self._status_txt2 = self._status.create_text(10, 40, anchor=Tkinter.NW, text=txt2,
                                                             font="Verdana 10 bold", fill="green")
                txt3 = "Notes in Queue:\n" + self._note_names_off()
                self._status_txt3 = self._status.create_text(10, 80, anchor=Tkinter.NW, text=txt3,
                                                             font="Verdana 10 bold", fill="green")

                self._canvas.delete(self._target_note_oval)
                staff_loc = self._trainer.current_target_note.staff_loc
                self._target_note_oval = self._canvas.create_oval(93, 106 - (5 * staff_loc),
                                                                  107, 114 - (5 * staff_loc),
                                                                  width=1, fill="green", outline="black")
                self._canvas.delete(self._sharp_flat_symbol)
                if len(self._trainer.current_target_note.name) > 1:
                    symbol = self._trainer.current_target_note.name[1:]
                    self._sharp_flat_symbol = self._canvas.create_text(90, 118 - (5 * staff_loc),
                                                                       font=tkFont.Font(family='Helvetica',
                                                                                        size=16),
                                                                       anchor=Tkinter.SE, text=symbol)

                self._in_training = False
                self._gui_top.after(1000, self._start_training)
            else:
                self._gui_top.after(50, self._check_note_practice_status)
        # print "Exiting _check_note_practice_status"

    def _on_set_topic(self, event):
        # set a new topic
        if event:
            pass
        value = self._training_topic_list.index(self._training_topic_list.curselection())
        self._trainer.current_topic = value

    def _on_threshold_change(self):
        self._trainer.threshold_multiplier = float(self._threshold_mult_spin.get())

    def _on_note_method_change_all(self):
        self._trainer.add_notes_method = self._trainer.ADD_NOTES_ALL_AT_ONCE

    def _on_note_method_change_inc(self):
        self._trainer.add_notes_method = self._trainer.ADD_NOTES_INCREMENTALLY

    def _on_closing(self):
        # kill the application
        self._trainer.kill_current_note_practice()  # used to kill the note capture thread.
        self._gui_top.destroy()

    def _on_add_notes_increment_change(self):
        self._trainer.add_notes_increment = int(self._add_notes_increment_spin.get())

    def _note_names_on(self):
        str_notes = ""
        for n in self._trainer.notes_in_play:
            if len(str_notes) > 0:
                str_notes += " "
            str_notes += n.name
        return str_notes

    def _note_names_off(self):
        str_notes = ""
        for n in self._trainer.notes_in_queue:
            if len(str_notes) > 0:
                str_notes += " "
            str_notes += n.name
        return str_notes


if __name__ == "__main__":
    gui = MainGUI()
