import os
import sys
from setuptools import setup, find_packages, Extension

sdkdir = 'vamp-plugin-sdk/src/vamp-hostsdk/'
vpydir = 'native/'

sdkfiles = [ 'Files', 'PluginBufferingAdapter', 'PluginChannelAdapter',
             'PluginHostAdapter', 'PluginInputDomainAdapter', 'PluginLoader',
             'PluginSummarisingAdapter', 'PluginWrapper', 'RealTime' ]
vpyfiles = [ 'PyPluginObject', 'PyRealTime', 'VectorConversion', 'vampyhost' ]

srcfiles = [
    sdkdir + f + '.cpp' for f in sdkfiles
] + [
    vpydir + f + '.cpp' for f in vpyfiles
]

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()



class get_numpy_include(str):
    def __str__(self):
        import numpy
        return numpy.get_include()

extra_compile_args = ['-fPIC', '-O2']

if sys.platform == 'darwin':
    extra_compile_args.extend(['-stdlib=libc++'])
elif sys.platform == 'linux':
    extra_compile_args.extend(['-fno-strict-aliasing'])

vampyhost = Extension('vampyhost',
                      language='c++',
                      sources = srcfiles,
                      define_macros = [ ('_USE_MATH_DEFINES', 1) ],
                      include_dirs = [ 'vamp-plugin-sdk', get_numpy_include()],
                      extra_compile_args=extra_compile_args)

setup (name = 'vamphost',
       version = '1.3.0',
       python_requires=">=3.9",
       url = 'https://code.soundsoftware.ac.uk/projects/vampy-host',
       description = 'Use Vamp plugins for audio feature analysis.',
       long_description = ( read('README.rst') + '\n\n' + read('COPYING.rst') ),
       license = 'MIT',
       packages = find_packages(exclude = [ '*test*' ]),
       ext_modules = [ vampyhost ],
       install_requires = ['numpy'],
       setup_requires = ['numpy'],
       author = 'Chris Cannam, George Fazekas',
       author_email = 'cannam@all-day-breakfast.com',
       classifiers = [
           'Development Status :: 4 - Beta',
           'Intended Audience :: Science/Research',
           'Intended Audience :: Developers',
           'License :: OSI Approved :: MIT License',
           'Operating System :: MacOS :: MacOS X',
           'Operating System :: Microsoft :: Windows',
           'Operating System :: POSIX',
           'Programming Language :: Python',
           'Programming Language :: Python :: 3',
           'Topic :: Multimedia :: Sound/Audio :: Analysis'
           ]
       )
