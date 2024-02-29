from midi_tokenizer import MidiBertTokenizer
# from Transformer import Transformer , simpleTransformer
import torch
from bert_midi import BertMidi
# from playground.fake_data_generator import fake_data_generator

if __name__ == '__main__':
    dataset = ['dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid']

    midi_bert_tokenizer = MidiBertTokenizer()
    midi_bert_tokenizer.tokenize_midi_dataset(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertMidi()
    model = model.to(device)

    #! test
    for _ in midi_bert_tokenizer.d_val_loader:
        x = _

    res = model.forward(x[0],x[2])

    print(res)

    model.train(data)

    print(model.predict(['PitchDrum_46 Velocity_127 Duration_0.1.8 Rest_0.1.8 Position_28 Program_-1 PitchDrum_44 Velocity_127 Duration_0.1.8 Program_-1 PitchDrum_42 Velocity_127 Duration_0.1.8 Rest_0.1.8 Position_30 Program_-1 PitchDrum_38 Velocity_127 Duration_0.1.8 Program_-1 PitchDrum_42 Velocity_127 Duration_0.1.8 Rest_0.1.8 Position_0 Program_-1 PitchDrum_42 Velocity_127 Duration_0.1.8 Rest_0.1.8']))

    # midi_tensor = tokenize_midi(file_path)
    # midi_tensor = torch.load("data/midi_tensor.torch")
    # train_dataloader , val_dataloader = fake_data_generator()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"[+] device {device}")
    # model = Transformer(
    #     num_tokens=4*2, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1, device=device
    # ).to(device)
    # train_loss_list, validation_loss_list = model.fit(train_dataloader, val_dataloader,10)

