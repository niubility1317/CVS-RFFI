"""Run the fixed source diagnostic that separates packet front and tail."""

from .preamble_component_source import main


if __name__ == '__main__':
    main(edge_split=True)
