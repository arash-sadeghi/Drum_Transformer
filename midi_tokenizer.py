import mido
import torch
from tqdm import tqdm
MIDI_LOOP_LIMIT = 30
# Load MIDI file and extract drum track
def load_midi(file_path):
    midi = mido.MidiFile(file_path)
    drum_track = []

    loop_limiter = 0
    # for msg in tqdm(midi.play()):
    for msg in midi.play():
        if msg.type == 'note_on' and msg.channel == 9:  # Drum channel
            drum_track.append(msg.note) #! the rest of midi information is diposed

        #! this part is just for observation
        print(f"[INFO] {loop_limiter} : {msg}")
        loop_limiter += 1 
        if loop_limiter>=MIDI_LOOP_LIMIT: break

    return drum_track

# Preprocess the data and convert it into PyTorch tensors
def midi2tensor(data):
    # Your preprocessing logic goes here
    # This is a simple example, you may need to create a vocabulary or other preprocessing steps
    return torch.tensor(data, dtype=torch.float32).view(-1, 1, 1)

# Train the transformer model
def train_transformer(data, model, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

#! main function
def tokenize_midi(file_path):
    drum_track = load_midi(file_path)
    midi_tensor = midi2tensor(drum_track)
    return midi_tensor

def test(file_path):
    #TODO tokenization is not completely exact but its good enough
    from miditok import REMI, TokenizerConfig  # here we choose to use REMI
    from pathlib import Path

    decode_midi_path = "hidden_velocity.midi"
    # Our parameters
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        # "num_velocities": 32,
        "num_velocities": 1,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_programs": True,
        "num_tempos": 32,  # number of tempo bins
        "tempo_range": (40, 250),  # (min, max)
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)

    # Creates the tokenizer
    tokenizer = REMI(config)

    # Tokenize a MIDI file
    tokens = tokenizer(Path("dataset/groove/drummer1/session1", "1_funk_80_beat_4-4.mid"))  # automatically detects Score objects, paths, tokens

    # Convert to MIDI and save it
    generated_midi = tokenizer(tokens)  # MidiTok can handle PyTorch/Numpy/Tensorflow tensors

    # Change the instrument type of the track
    # for track in tokens:
    #     for event in track:
    #         if event.type == 'program_change':
    #             event.value = 1 # Change the instrument type here

    generated_midi.dump_midi(Path(".", decode_midi_path))
    print(f"{'_'*10} content of detokinized {'_'*10}")
    load_midi(decode_midi_path)

# Example usage
if __name__ == "__main__":
    file_path = 'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    midi_tensor = tokenize_midi(file_path)
    test(file_path)


