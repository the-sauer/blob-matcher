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


def flip(f):
    def flipped_f(y, x):
        return f(x, y)
    return flipped_f


def fchain(*args):
    def function(x):
        for f in reversed(args):
            x = f(x)
        return x
    return function


def curry(f):
    def function(x):
        def inner(y):
            return f(x,y)
        return inner
    return function
