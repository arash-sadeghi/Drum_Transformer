import mido
import torch
from tqdm import tqdm
from miditok import REMI, TokenizerConfig  # here we choose to use REMI
import random
import pandas as pd

MIDI_LOOP_LIMIT = 30
WINDOW_SIZE = 30

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


def put_tokens2windows(input_list , window_size = WINDOW_SIZE):
    return [' '.join(input_list[i:i + window_size]) for i in range(0, len(input_list), window_size)]

def shuffle_in_same_manner(list1, list2):
    # Combine the lists element-wise
    combined_lists = list(zip(list1, list2))

    # Shuffle the combined lists
    random.shuffle(combined_lists)

    # Unzip the shuffled lists
    shuffled_list1, shuffled_list2 = zip(*combined_lists)

    return list(shuffled_list1), list(shuffled_list2)

def tokenize_midi(file_path):
    # drum_track = load_midi(file_path)
    # midi_tensor = midi2tensor(drum_track)
    # return midi_tensor

    #TODO tokenization is not completely exact but its good enough

    decode_midi_path = "only_toks.midi"
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

    config_input = TokenizerConfig(**TOKENIZER_PARAMS)

    # Creates the tokenizer
    tokenizer_input = REMI(config_input)

    # Tokenize a MIDI file
    tokens_input = tokenizer_input(file_path)  # automatically detects Score objects, paths, tokens

    token_input_windowed = put_tokens2windows(tokens_input.tokens)

    TOKENIZER_PARAMS["num_velocities"] = 32
    config_output = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer_output = REMI(config_output)
    tokens_output = tokenizer_output(file_path)  # automatically detects Score objects, paths, tokens
    token_output_windowed = put_tokens2windows(tokens_output.tokens)

    token_input_windowed_shuffled , token_output_windowed_shuffled = shuffle_in_same_manner(token_input_windowed , token_output_windowed)

    data = pd.DataFrame({'input_text': token_input_windowed_shuffled, 'target_text': token_output_windowed_shuffled})

    return data

    # # Convert to MIDI and save it
    # generated_midi = tokenizer(tokens.tokens)  # MidiTok can handle PyTorch/Numpy/Tensorflow tensors
    # generated_midi.dump_midi(decode_midi_path)
    # print(f"{'_'*10} content of detokinized {'_'*10}")
    # load_midi(decode_midi_path)
    # print("hi")

# Example usage
if __name__ == "__main__":
    # file_path = 'dataset/groove/drummer6/session2/2_rock_95_beat_4-4.mid'#'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    file_path = 'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    midi_tensor = tokenize_midi(file_path)


