import nox


@nox.session
def tests(session):
    session.install("pip", "numpy", "pytest", "matscipy")
    session.run("pip", "install", ".", "-v")
    session.run("pytest")
