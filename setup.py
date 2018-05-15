from setuptools import setup

setup(name="streamvis",
      version="0.2.1",
      author="Ivan Usov",
      author_email="ivan.usov@psi.ch",
      description="Stream visualization of JF data.",
      packages=['streamvis'],
      license='GNU GPLv3',
      requires=['bokeh',
                'colorcet',
                'numpy',
                'pyzmq',
                'matplotlib',
                'hdf5plugin',
                'h5py',
                'pillow',
                'tornado',
               ],
     )
