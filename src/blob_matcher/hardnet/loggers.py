# Copyright 2019 EPFL, Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


class FileLogger:
    "Log text in file."

    def __init__(self, path):
        self.path = path

    def log_string(self, file_name, string):
        """Stores log string in specified file."""
        text_file = open(self.path + file_name + ".log", "a")
        text_file.write(string + '' + str(string) + '\n')
        text_file.close()

    def log_stats(self, file_name, text_to_save, value):
        """Stores log in specified file."""
        text_file = open(self.path + file_name + ".log", "a")
        text_file.write(text_to_save + ' ' + str(value) + '\n')
        text_file.close()
