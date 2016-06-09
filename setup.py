from distutils.core import setup

setup(
    name='SuPyLearner',
    version='0.1.1',
    author='Sam Lendle, Alex Keil',
    description='Implementation of the SuperLearner algorithm',
    author_email='akeil@unc.edu',
    packages=['supylearner'],
    url='https://github.com/alexpkeil1/SuPyLearner',
    license='GPL-3',
    long_description=open('README').read(),
    requires=[
        "scipy (>= 0.10.1)",
        "numpy (>= 1.6)",
#        "scikit-learn (>= 0.10)", #distutils doesn't seem to like the - in scikit-learn
    ],
)
