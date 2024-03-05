import lyricsgenius
import re

genius = lyricsgenius.Genius("wTiJ3xmBXTwXXio7xtRzuWFJUP09WhBYbyelDP60gW2euJSM8jMrIIRqoAJVzrwg")
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = False
genius.excluded_terms = ["(Remix)", "(Live)"]

artists = []

with open("listofsingers.txt", "r", encoding="utf-8") as file:
    for line in file:
        artist_name = line.strip()
        artist = genius.search_artist(artist_name)
        print(artist.songs)
        if artist is not None:
            artists.append(artist)
        else:
            print(f"No information found for '{artist_name}'")

with open("all_lyrics.txt", "w", encoding="utf-8") as file:
    for artist in artists:
        for song in artist.songs:
            song_lyrics = genius.search_song(song.title, artist.name)
            if song_lyrics is not None:
                lines = song_lyrics.lyrics.split("Lyrics")[1]
                lines = lines.split("You might")[0]
                pattern = r"\((.*?)\)"
                result = re.sub(pattern, "", lines)
                modified_lines = result.replace("You might also like", "").strip()
                file.write(f"Artist: {artist.name}\n")
                file.write(f"Song: {song.title}\n")
                file.write(modified_lines)
                file.write("\n\n\n")
            else:
                print(f"No lyrics found for '{song.title}' by '{artist.name}'")

print("Lyrics saved to 'all_lyrics.txt'")
