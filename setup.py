from distutils.core import setup
import imp
import os

setup_requires = []
install_requires = [
    'chainer>=2.0',
]


here = os.path.abspath(os.path.dirname(__file__))
__version__ = imp.load_source(
    '_version', os.path.join(here,
                             'python/chainer_trt',
                             '_version.py')).__version__

setup(name='chainer-trt',
      version=__version__,
      description='chainer-trt: Chainer x TensorRT',
      author='',
      author_email='',
      packages=['chainer_trt'],
      package_dir={'': 'python'},
      license='',
      # url='',
      setup_requires=setup_requires,
      install_requires=install_requires
      )
