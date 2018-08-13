def main():
    """
    Execute the "streamvis" command line program.

    This is a wrapper around 'bokeh serve' command which provides a user interface to launch
    applications bundled with the streamvis package.

    For more information, see:
    https://bokeh.pydata.org/en/latest/docs/reference/command/subcommands/serve.html
    """

    import os
    import sys
    import subprocess

    apps_path = os.path.join(os.environ['CONDA_PREFIX'], 'streamvis-apps')

    # TODO: generalize streamvis parsing after python/3.7 release
    # due to an issue with 'argparse' (https://bugs.python.org/issue14191),
    # which is supposed to be fixed in python/3.7, keep parsing unflexible, but very simple
    _, app_name, *app_args = sys.argv

    command = ['bokeh', 'serve', os.path.join(apps_path, app_name), *app_args]
    print(' '.join(command))
    subprocess.run(command)

if __name__ == "__main__":
    main()
