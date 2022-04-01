#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import string
import re
import os
import sqlite3
import subprocess

import chardet
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pymorphy2

from pysubparser import parser as subParser

# Init language toolkits
lemmatize = WordNetLemmatizer().lemmatize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
morph = pymorphy2.MorphAnalyzer()


def extract_frame(data):
  """Extracts a frame from a video file

  Args:
    data: A dict containing:
      'vid': path of video file
      'timestamp': timestamp of a frame to extract
      'out': path of output file

  Returns:
    Path of the extracted frame
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


def analyze(data):
  """Analyzes data and returns a list of lemmatized tokens

  Args:
    data: A plain string of text to analyze

  Returns:
    result: A list of tokens
  """

  # Tokenize lowecase text and remove punctuation
  words = word_tokenize(
      re.sub(f'[{string.punctuation}]+', '', data.lower()))

  # Lemmatize words
  result = []
  for word in words:
    lang = chardet.detect(word.encode('cp1251'))['language']
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


def parse_subs(path, globs, conn):
  """Finds video files and matched subtitles in 'path' and returns subtitle events

  A subtitle is considered matched to a video file if ones file name
  without file extensions is a subset of the subtitle file name.

  Args:
    path: PosixPath, where to start looking for files
    globs: A dictionary of globs to search for.
      'video': is a list of valid video file globs
      'subtitle': is a list of valid subtitle file globs
    conn: A sqlite3 connection

  Returns:
    A list of dicts, containing:
    'timestamp': datetime of a subtitle event
    'text': key with the event text
    'sub': path of subtitle file
    'vid': path of video file

  Raises:
    FileNotFoundError: If no video or subtitle files are found
  """

  sub_files = []
  for ext in globs["subtitle"]:
    sub_files.extend(path.glob(ext))
  # remove sub_files from sub_files if it was found in conn database
  for sub in sub_files:
    conn.execute('''SELECT * FROM events WHERE sub_path = ?''', (sub,))
    if conn.fetchone():
      sub_files.remove(sub)

  vid_files = []
  for ext in globs["video"]:
    vid_files.extend(path.glob(ext))

  if not vid_files:
    raise FileNotFoundError('No video files found')
  if not sub_files:
    raise FileNotFoundError('No subtitle files found')

  events = []
  vid_sub_match = ''
  for sub_file in sub_files:
    for vid_file in vid_files:
      if sub_file.find(
              vid_file.name.split('.')[0]) != -1:
        vid_sub_match = vid_file
        break
    if not vid_sub_match:
      break

    subtitle = subParser.parse(str(sub_file))
    for event in subtitle:
      events.append({
          'timestamp': event.end - (event.end - event.start) / 2,
          'text': event.text,
          'sub': sub_file,
          'vid': vid_sub_match
      })

  return events


def main():
  """Main function"""

  # Add script arguments
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

  args = vars(parser.parse_args())

  # Check if path exists
  if not Path.exists(args['path'][0]):
    raise FileNotFoundError(
        'Path {} does not exist'.format(args['path'][0]))

  # Change working directory to script's path
  os.chdir(args['path'][0])

  # Create database for storing subtitle events
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
                    VALUES (?, ?, ?, ?)''', (event['timestamp'].strftime('%H:%M:%S'),
                                             ' '.join(analyze(event['text'])),
                                             event['sub'],
                                             event['vid']))


if __name__ == "__main__":
  main()
