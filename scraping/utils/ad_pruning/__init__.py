from lxml.etree import tostring
import lxml.html
from pathlib import Path
from .adremover import AdRemover

_dir = Path("utils/ad_pruning/rules/")
_remover = AdRemover(_dir / "adblockplus.txt", _dir / "adtidy.txt")


def remove_ads(html: lxml.etree.ElementTree):
    return _remover.remove_ads(html)
