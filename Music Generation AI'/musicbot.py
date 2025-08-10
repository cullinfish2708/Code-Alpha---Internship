import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from music21 import converter, instrument, note, chord, stream
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pretty_midi
import fluidsynth
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class MusicGenerator:
    def __init__(self, data_path='midi_files/', seq_length=100):
        self.data_path = data_path
        self.seq_length = seq_length
        self.notes = []
        self.vocab = None
        self.n_vocab = 0
        self.network_input = None
        self.network_output = None
        self.model = None
        
    def load_midi_files(self):
        """Load and parse MIDI files from directory"""
        print(f"Loading MIDI files from {self.data_path}...")
        midi_files = []
        
        # Supported MIDI file extensions
        extensions = ('*.mid', '*.midi', '*.MID', '*.MIDI')
        for ext in extensions:
            midi_files.extend(glob.glob(os.path.join(self.data_path, ext)))
            
        if not midi_files:
            raise FileNotFoundError(f"No MIDI files found in {self.data_path}")
            
        print(f"Found {len(midi_files)} MIDI files")
        
        # Parse files with error handling
        for i, file in enumerate(midi_files):
            try:
                midi = converter.parse(file)
                print(f"Processing file {i+1}/{len(midi_files)}: {os.path.basename(file)}")
                
                # Get all notes and chords
                parts = instrument.partitionByInstrument(midi)
                if parts:
                    notes_to_parse = parts.parts[0].recurse()
                else:
                    notes_to_parse = midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        self.notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        self.notes.append('.'.join(str(n) for n in element.normalOrder))
                    elif isinstance(element, note.Rest):
                        self.notes.append('Rest')
                        
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        if not self.notes:
            raise ValueError("No notes extracted from MIDI files")
            
        print(f"Successfully extracted {len(self.notes)} musical events")
        return self.notes
    
    def create_vocabulary(self):
        """Create vocabulary of unique musical events"""
        unique_events = sorted(set(self.notes))
        self.vocab = dict((event, number) for number, event in enumerate(unique_events))
        self.n_vocab = len(unique_events)
        print(f"Created vocabulary of {self.n_vocab} unique musical events")
        return self.vocab
    
    def prepare_sequences(self):
        """Prepare input and output sequences for training"""
        if not self.notes or not self.vocab:
            raise RuntimeError("Load MIDI files and create vocabulary first")
            
        print("Preparing training sequences...")
        
        # Create input sequences and corresponding outputs
        input_sequences = []
        output_sequences = []
        
        for i in range(0, len(self.notes) - self.seq_length, 1):
            sequence_in = self.notes[i:i + self.seq_length]
            sequence_out = self.notes[i + self.seq_length]
            input_sequences.append([self.vocab[char] for char in sequence_in])
            output_sequences.append(self.vocab[sequence_out])
        
        n_patterns = len(input_sequences)
        print(f"Created {n_patterns} training sequences")
        
        # Reshape and normalize input
        self.network_input = np.reshape(input_sequences, (n_patterns, self.seq_length, 1))
        self.network_input = self.network_input / float(self.n_vocab)
        
        # One-hot encode output
        self.network_output = tf.keras.utils.to_categorical(output_sequences)
        
        return self.network_input, self.network_output
    
    def build_model(self, lstm_units=512, dropout_rate=0.3):
        """Build LSTM model architecture"""
        print("Building LSTM model...")
        
        self.model = Sequential([
            LSTM(lstm_units, input_shape=(self.network_input.shape[1], self.network_input.shape[2]), 
            Dropout(dropout_rate),
            Dense(lstm_units, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(self.n_vocab, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy', 
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        return self.model
    
    def train_model(self, epochs=100, batch_size=64, validation_split=0.2):
        """Train the model with callbacks and validation"""
        if not self.model:
            raise RuntimeError("Build model first")
            
        # Create callbacks
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        print("Starting model training...")
        history = self.model.fit(
            self.network_input, 
            self.network_output,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint, early_stop]
        )
        
        # Plot training history
        self.plot_training_history(history)
        return history
    
    def plot_training_history(self, history):
        """Plot training and validation loss/accuracy"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def generate_music(self, start_sequence=None, length=500, temperature=0.7):
        """Generate new music sequence"""
        if not self.model:
            self.model = tf.keras.models.load_model('best_model.h5')
            
        if not self.vocab:
            raise RuntimeError("Vocabulary not available")
            
        # Reverse vocabulary for decoding
        rev_vocab = {i: event for event, i in self.vocab.items()}
        
        # Use random start sequence if not provided
        if start_sequence is None:
            start_idx = np.random.randint(0, len(self.network_input)-1)
            start_sequence = self.network_input[start_idx]
        
        pattern = list(start_sequence.flatten() * float(self.n_vocab))
        output = []
        
        print("Generating new music sequence...")
        for _ in range(length):
            # Prepare input
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)
            
            # Make prediction
            prediction = self.model.predict(prediction_input, verbose=0)[0]
            
            # Apply temperature sampling
            prediction = np.log(prediction) / temperature
            exp_preds = np.exp(prediction)
            probabilities = exp_preds / np.sum(exp_preds)
            
            # Sample from distribution
            index = np.random.choice(range(self.n_vocab), p=probabilities)
            result = rev_vocab[index]
            output.append(result)
            
            # Update pattern
            pattern.append(index)
            pattern = pattern[1:]
        
        return output
    
    def create_midi(self, prediction_output, filename='generated_music.mid'):
        """Convert generated sequence to MIDI file"""
        print(f"Creating MIDI file: {filename}")
        
        offset = 0
        output_notes = []
        
        for event in prediction_output:
            # Handle notes
            if (event == 'Rest') or (event not in self.vocab):
                new_note = note.Rest()
            elif '.' in event:
                notes_in_chord = event.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(event)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            
            # Increase offset for next event
            offset += 0.5
        
        # Create MIDI stream
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=filename)
        print(f"MIDI file saved as {filename}")
        return filename
    
    def play_midi(self, midi_file):
        """Play MIDI file using fluidsynth"""
        try:
            # Load SoundFont (download one if needed)
            soundfont = 'soundfont.sf2'
            if not os.path.exists(soundfont):
                print("Downloading SoundFont...")
                import urllib.request
                url = "https://www.dropbox.com/s/vfr8t4x1a9d1xqz/GeneralUserGS.sf2?dl=1"
                urllib.request.urlretrieve(url, soundfont)
            
            # Initialize synthesizer
            fs = fluidsynth.Synth()
            fs.start()
            sfid = fs.sfload(soundfont)
            fs.program_select(0, sfid, 0, 0)
            
            # Play MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    fs.noteon(0, note.pitch, 100)
                    time.sleep(note.duration)
                    fs.noteoff(0, note.pitch)
            
            # Cleanup
            time.sleep(1)  # Let notes finish
            fs.delete()
            return True
            
        except Exception as e:
            print(f"Error playing MIDI: {str(e)}")
            return False

# Main execution
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'data_path': 'midi_files/classical',
        'seq_length': 100,
        'epochs': 150,
        'batch_size': 128,
        'generation_length': 300,
        'temperature': 0.7,
        'output_file': 'generated_composition.mid'
    }
    
    try:
        # Initialize music generator
        generator = MusicGenerator(
            data_path=CONFIG['data_path'],
            seq_length=CONFIG['seq_length']
        )
        
        # Load and process data
        generator.load_midi_files()
        generator.create_vocabulary()
        generator.prepare_sequences()
        
        # Build and train model
        generator.build_model(lstm_units=512, dropout_rate=0.3)
        generator.train_model(
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size']
        )
        
        # Generate new music
        new_music = generator.generate_music(
            length=CONFIG['generation_length'],
            temperature=CONFIG['temperature']
        )
        
        # Save as MIDI file
        midi_file = generator.create_midi(
            new_music,
            filename=CONFIG['output_file']
        )
        
        # Play the generated composition
        print("Playing generated composition...")
        generator.play_midi(midi_file)
        
    except Exception as e:
        print(f"Error in music generation pipeline: {str(e)}")