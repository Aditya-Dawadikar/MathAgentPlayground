def pytest_addoption(parser):
    parser.addoption(
        "--agent-module",
        action="store",
        default=None,
        help="Python module path for the agent under test, for example app.agent or app.agent_v2.",
    )