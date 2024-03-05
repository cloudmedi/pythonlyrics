import lyricsgenius
import re

genius = lyricsgenius.Genius("wTiJ3xmBXTwXXio7xtRzuWFJUP09WhBYbyelDP60gW2euJSM8jMrIIRqoAJVzrwg")
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = False
genius.excluded_terms = ["(Remix)", "(Live)"]


artist = genius.search_artist("sezen aksu", max_songs=3, sort="title")
artist.save_lyrics()