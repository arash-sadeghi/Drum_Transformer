import mido
import torch
from tqdm import tqdm

# Load MIDI file and extract drum track
def load_midi(file_path):
    midi = mido.MidiFile(file_path)
    drum_track = []

    for msg in tqdm(midi.play()):
        if msg.type == 'note_on' and msg.channel == 9:  # Drum channel
            drum_track.append(msg.note) #! the rest of midi information is diposed

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

# Example usage
if __name__ == "__main__":
    file_path = 'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    midi_tensor = tokenize_midi(file_path)
    print(midi_tensor)


