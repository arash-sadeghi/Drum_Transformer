import mido
import torch
from tqdm import tqdm
from miditok import REMI, TokenizerConfig  # here we choose to use REMI
import random
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os 

MIDI_LOOP_LIMIT = 30
WINDOW_SIZE = 30
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

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

def put_ids2str(id_list):
    return ' '.join(str(_) for _ in id_list)

def shuffle_in_same_manner(list1, list2):
    # Combine the lists element-wise
    combined_lists = list(zip(list1, list2))

    # Shuffle the combined lists
    random.shuffle(combined_lists)

    # Unzip the shuffled lists
    shuffled_list1, shuffled_list2 = zip(*combined_lists)

    return list(shuffled_list1), list(shuffled_list2)

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Add any preprocessing steps here if needed

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve data at the given index
        sample = self.dataframe.iloc[idx]
        # Convert data to PyTorch tensors (if needed)
        # For example:
        data = torch.tensor(sample['input'])
        label = torch.tensor(sample['target'])
        attention_mask = torch.tensor(sample['attention_mask'])

        # Return a tuple (or dictionary) containing the data and label
        return data, label, attention_mask


class MidiBertTokenizer:
    BATCH_SIZE = 16
    RANDOM_SEED = 100
    DICT_FILE_URL = os.path.join("data","data_dict","data_dict.json")
    SCALING_FACTOR = 127
    def __init__(self) -> None:

        
        self.inp_tgt = {'input':[], 'attention_mask':[],'target':[]}

        TOKENIZER_PARAMS = {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 32,
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
        self.tokenizer_input = REMI(config_input)

        self.bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def tokenize_midi_dataset(self,file_list): #! expected to recieve a list of paths corrsponding to midi data
        print("[+] loading midi files to data_loader")
        if os.path.exists(MidiBertTokenizer.DICT_FILE_URL):
            print(f"[+] loading data dictionary from file {MidiBertTokenizer.DICT_FILE_URL}")
            with open(MidiBertTokenizer.DICT_FILE_URL, 'r') as f:
                self.inp_tgt = json.load(f)

        else:
            print(f"[+] writing data to dictionary file {MidiBertTokenizer.DICT_FILE_URL}")

            for file_path in tqdm(file_list):
                self.tokenize_midi_file(file_path)
            
            with open(MidiBertTokenizer.DICT_FILE_URL, 'w') as f:
                json.dump(self.inp_tgt, f)
        
        
        self.generate_data_loader()    

    def tokenize_midi_file(self,file_path):
        try: #! some midi files are not structured well
            midi_tokens = self.tokenizer_input(file_path)  # automatically detects Score objects, paths, tokens
        except Exception as e:
            print(f"[-] Error reading {file_path}, Error: {e}")
            return
        
        token_length = len(midi_tokens.ids)
        
        target_velocity = []
        input_midi_ids = []
        for i in range(token_length): #! hiding velocity information in target
            if 'Velocity' in midi_tokens.tokens[i]:
                input_midi_ids.append('[MASK]')

                velocity = int(midi_tokens.tokens[i].split('_')[1])
                velocity = velocity/MidiBertTokenizer.SCALING_FACTOR #! normalizing

                assert velocity>0 and velocity<=1

                target_velocity.append(velocity)

            else:
                input_midi_ids.append(str(midi_tokens.ids[i]))
            

        
        #! make input a single string
        input_str = ' '.join(_ for _ in input_midi_ids)

        inp_bert_tokens = self.bert_tokenizer(input_str)

        vel_loop_counter = 0
        for c in range(len(inp_bert_tokens.input_ids)//510):
            inp_bert_tokenized = inp_bert_tokens.input_ids[c*510:(c+1)*510]
            
            target_velocity_equal_len = [0]*len(inp_bert_tokenized)

            for _ in inp_bert_tokenized:
                if _ == 103:
                    target_velocity_equal_len[c] = target_velocity[vel_loop_counter]

            vel_loop_counter +=1
            self.inp_tgt['input'].append(inp_bert_tokenized)  #! 510 is bert capacity
            self.inp_tgt['attention_mask'].append(inp_bert_tokens.attention_mask[c*510:(c+1)*510])  #! 510 is bert capacity
            self.inp_tgt['target'].append(target_velocity_equal_len)  #! 510 is bert capacity

    def generate_data_loader(self):
        data = pd.DataFrame(self.inp_tgt)


        df_train, df_test = train_test_split(data, test_size=0.2, random_state=MidiBertTokenizer.RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=MidiBertTokenizer.RANDOM_SEED)

        # Create DataLoader for each set
        self.d_train_loader = DataLoader(CustomDataset(df_train), batch_size=MidiBertTokenizer.BATCH_SIZE, shuffle=True)
        self.d_test_loader = DataLoader(CustomDataset(df_test), batch_size=MidiBertTokenizer.BATCH_SIZE, shuffle=True)
        self.d_val_loader = DataLoader(CustomDataset(df_val), batch_size=MidiBertTokenizer.BATCH_SIZE, shuffle=True)


# Example usage
if __name__ == "__main__":
    # file_path = 'dataset/groove/drummer6/session2/2_rock_95_beat_4-4.mid'#'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    file_path = 'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    midi_tensor = tokenize_midi(file_path)


