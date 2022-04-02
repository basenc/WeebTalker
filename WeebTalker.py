#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Look for subtitle line matching
your search query and renders it to annoy your friends with
tons of anime screenshots instead of talking
like a human being

Example:
    ``python WeebTalker.py -p torrents/animes/SAO_S1_1080p[AniDub] -q 'baka'``

Todo:
    * Better search function
"""

import argparse
from pathlib import Path
import string
import re
import os
import sqlite3
import subprocess
import logging

import chardet
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pymorphy2

from pysubparser import parser as subParser

import regex

# Init language toolkits
lemmatize = WordNetLemmatizer().lemmatize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

morph = pymorphy2.MorphAnalyzer()


def convert_ms_to_ffmpeg_seek(ms):
  ms = int(ms)
  h = ms // 3600000
  m = (ms % 3600000) // 60000
  s = (ms % 60000) // 1000
  ms = ms % 1000
  return f'{h}:{m}:{s}.{ms}'


def extract_frame(data) -> Path:
  """Extracts a frame from a video file

  Args:
    data (dict): A dictionary containing:
    {vid (pathlib.Path): path of video file
    sub (pathlib.Path): path of subtitle file
    timestamp (str): datetime of a subtitle event}

  Returns:
    pathlib.Path: Path to the extracted frame
  """

  ffmpeg_cmd = f'ffmpeg -ss {data["timestamp"]} -i "{data["vid"]}" -copyts -vf ass="{data["sub"]}" -vframes 1 -y /tmp/frame.png'
  subprocess.run(ffmpeg_cmd, shell=True, check=True)
  return Path('/tmp/frame.png')


def hline():
  """Prints a horizontal line 😐"""

  try:
    termwidth, _ = os.get_terminal_size()
  except OSError:
    termwidth = 32
  print('\n' + '=' * termwidth + '\n')


def analyze(data) -> list:
  """Analyzes data and returns a list of lemmatized tokens

  Args:
    data (str): A string of text to analyze

  Returns:
    result (list): A list of tokens
  """

  # remove non latin characters
  data = ''.join(re.findall(r'[\u0020-\u007F\u00A0-\u00FF\u0100-\u017F\u0180-\u024F]+', data))
  # remove .ass styles
  data = re.sub(r'\\[a-z]{1,3}\d{5,6}', '', data)
  data = re.sub(r'\{.*\}', '', data)
  # remove punctuation, lowecase and tokenize
  data = data.translate(str.maketrans('', '', string.punctuation)).lower()
  words = word_tokenize(data)

  # Lemmatize tokenized words
  result = []
  for word in words:
    # TODO: add proper multilang
    # lang = chardet.detect(word.encode('cp1251'))['language']
    lang = 'EN'
    if lang == 'Russian':
      if word not in stopwords.words('russian'):
        result.append(morph.normal_forms(word)[0])
    else:
      if word not in stopwords.words('english'):
        result.append(lemmatize(word))
  return result


# def quickSearchEngine(data, query):
#   analyzed_query = analyze(query)

#   analyzed_data = []
#   for line in enumerate(data):
#     analyzed_line = analyze(line[1])
#     if analyzed_line.issuperset(analyzed_query):
#       analyzed_data.append(analyzed_line, line[0])

#   return analyzed_data


def parse_subs(path, globs, conn) -> dict:
  """Finds video files and matched subtitles in 'path' and returns subtitle events

  A subtitle is considered matched to a video file if ones file name
  without file extensions is a subset of the subtitle file name.

  Args:
    path (pathlib.Path): where to start looking for files
    globs (dict): A dictionary of globs to search for. Consists of:

      * **video** (*pathlib.Path*): is a list of valid video file globs
      * **subtitle** (*pathlib.Path*): is a list of valid subtitle file globs
      * **conn** (*object*): A sqlite3 connection

  Returns:
    dict: A dictionary of subtitle events, containing:

      * **timestamp** (*int*): datetime of a subtitle event
      * **text** (*str*): key with the event text
      * **sub** (*pathlib.Path*): path of subtitle file
      * **vid** (*pathlib.Path*): path of video file

  Raises:
    FileNotFoundError: If no video or subtitle files are found
  """

  sub_files = []
  for ext in globs["subtitle"]:
    sub_files.extend(path.glob(ext))
  # remove sub_files from sub_files if it was found in conn database
  for sub in sub_files:
    if conn.execute('''SELECT * FROM events WHERE sub_path = ?''', (sub.as_posix(),)).fetchone():
      sub_files.remove(sub)

  vid_files = []
  for ext in globs["video"]:
    vid_files.extend(path.glob(ext))

  if not vid_files or not sub_files:
    logging.log(logging.CRITICAL, f'No video or subtitle files found in {path}')
    raise FileNotFoundError(
        f'No video or subtitle files are found in {path}')

  events = []
  vid_sub_match = ''
  for sub_file in sub_files:
    for vid_file in vid_files:
      if sub_file.as_posix().find(
              vid_file.name.split('.')[0]) != -1:
        logging.log(logging.INFO, f'Found subtitle {sub_file} for video {vid_file}')
        vid_sub_match = vid_file
        break
    if not vid_sub_match:
      break

    subtitle = subParser.parse(str(sub_file))
    for event in subtitle:
      events.append({
          'timestamp': event.end.microsecond - (event.end.microsecond - event.start.microsecond) / 2,
          'text': event.text,
          'sub': sub_file,
          'vid': vid_sub_match
      })

  return events


def main():
  parser = argparse.ArgumentParser(description='''
    Look for subtitle line matching
    your search query and renders it to annoy your friends with
    tons of anime screenshots instead of talking
    like a human being''')

  parser.add_argument('-p', '--path',
                      nargs=1,
                      type=Path,
                      help='Path where all of your shit\'s stored')

  parser.add_argument('-q', '--query',
                      nargs='?',
                      type=str,
                      default='',
                      help='Search query')

  parser.add_argument('-d', '--debug',
                      action='store_true',
                      help='Enable debug logging')

  if parser.parse_args().debug:
    logging.basicConfig(level=logging.DEBUG)

  args = vars(parser.parse_args())

  if not Path.exists(args['path'][0]):
    raise FileNotFoundError(f'Path {args["path"][0]} does not exist')

  # Change working directory to 'path'
  os.chdir(args['path'][0])

  # Init logging
  logging.basicConfig(filename='WeebTalker.log', level=logging.INFO)

  conn = sqlite3.connect('database.db')
  conn.execute('''CREATE TABLE IF NOT EXISTS events
                  (timestamp INT, text TEXT, sub_path TEXT, vid_path TEXT)''')

  # Valid video and subtitle globs for parse_subs funciton
  globs = {
      'video': ['**/*.mkv'],
      'subtitle': ['**/*.ass']
  }

  if not args['query'][0]:
    args['query'] = input('Enter your search query: ')

  for event in parse_subs(args['path'][0], globs, conn):
    conn.execute('''INSERT INTO events (timestamp, text, sub_path, vid_path)
                    VALUES (?, ?, ?, ?)''',
                 (
                     convert_ms_to_ffmpeg_seek(event['timestamp']),
                                             ' '.join(analyze(event['text'])),
                     event['sub'].as_posix(),
                     event['vid'].as_posix()
                 )
                 )

  conn.commit()
  conn.close()


if __name__ == "__main__":
  main()
