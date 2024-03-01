import os

def find_midi_files(root_path):
    midi_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith('.mid') or filename.lower().endswith('.midi'):
                # midi_files.append(os.path.relpath(os.path.join(dirpath, filename), root_path))
                midi_files.append(os.path.join(dirpath, filename))

    return midi_files
