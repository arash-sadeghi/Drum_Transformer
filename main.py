from midi_tokenizer import tokenize_midi
from Transformer import Transformer
import torch
from playground.fake_data_generator import fake_data_generator

if __name__ == '__main__':
    file_path = 'dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    midi_tensor = tokenize_midi(file_path)
    # midi_tensor = torch.load("data/midi_tensor.torch")

    train_dataloader , val_dataloader = fake_data_generator()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] device {device}")

    model = Transformer(
        num_tokens=4*2, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1, device=device
    ).to(device)


    train_loss_list, validation_loss_list = model.fit(train_dataloader, val_dataloader,10)

