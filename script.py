import os

dataset_path = "/Users/xenios/Documents/MusicGenreClassification/dataset"

# Check genres (subfolders)
genres = [g for g in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, g))]
print("Genres:", genres)

# Check number of files per genre
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    num_files = len(os.listdir(genre_path))
    print(f"Genre: {genre}, Number of files: {num_files}")
