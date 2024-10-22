try: from pip._internal.operations import freeze
except ImportError: # pip < 10.0
    from pip.operations import freeze

pkgs = freeze.freeze()
with open("requirements.txt", "a") as arquivo:
    for pkg in pkgs:
        print(pkg)
        arquivo.write(pkg+"\n")
