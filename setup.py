from distutils.core import setup


setup(
    name='poprank',
    version='1.0.0',
    description='Rating Mechanisms',
    author='Manfred Diaz, Aurelien Buck-Kaeffer',
    author_email='',
    url='https://www.python.org/sigs/distutils-sig/',
    package_dir={'': 'src'},
    packages=['poprank'],
    install_requires=[
        'popcore @ git+ssh://git@github.com/poprl/popcore@v1.0.0-beta#egg=popcore',
    ]
)
