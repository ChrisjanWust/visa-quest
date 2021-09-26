from enum import Enum


class PageType(Enum):
    LIST = "list"
    AD = "ad"
    NA = "na"


def derive_page_type(url: str) -> PageType:
    """
    This should work for:
    - indeed
    - careerjet
    """
    rel_url = url.split("/", 1)[1].lower()
    if any(
        keyword in rel_url
        for keyword in ["jobad", "viewjob", "adverts", "jobs/detail/"]
    ):
        return PageType.AD
    if "monster.co.uk/job-openings/" in url:
        return PageType.AD
    if (
        "search" in rel_url
        or "jobs" in rel_url
        or len(rel_url) < len("search/jobs?s=&l=USA")
    ):
        return PageType.LIST
    return PageType.NA


def test_page_type():
    data = [
        (
            "https://www.careerjet.com/jobad/us35ac603febdf46e4e5add735afd1fe31",
            PageType.AD,
        ),
        (
            "https://www.simplyhired.com/search?q=python+nlp+visa++sponsorship&l=&job=oPvjudQ95-jxco46_1sVPfx3rAD8zZeSkAsT0qhXY7h_5fZi7nmr0Q",
            PageType.LIST,
        ),
        (
            "https://za.indeed.com/viewjob?jk=92c290b77cb98c46&l=Cape+Town%2C+Western+Cape&tk=1fgeigkf9r29r802&from=web&advn=1108560342950489&adid=13700033&ad=-6NYlbfkN0CMw59liI42PwttBjy-9jar_84QrIT-43CZ2mQD3J1LR6Hyel2mXxXo7NsilUnjzM0_1SVvG5ES4FaSHN4ea6SNkc6g_8_TEFEJMbbAhXortIqeAfEWtx1QA8jcU3rmNXH4G8B4w_bXDLv5zqDPREz6P_Dgr_5AEXPexWvHHzRo1tnqJMzwyGFAy5w9kRnzfGtAAneOpf0znro_L_mHVbTE6APsbDrsI1AHp6Wuj0JETGPutcW-Th0j3qsNVJ9EKaQYRSgiP_P0vAsM7A6kvC-ecTor6fvglc1l35rW2cxL6VC24m8aaA1xTdsLEYUCLW7nXpVgZjmhitVn1R0KMOA18NvB2u5NlvGBseLvaoPE8A%3D%3D&pub=4a1b367933fd867b19b072952f68dceb&vjs=3",
            PageType.AD,
        ),
        (
            "https://za.indeed.com/jobs?q&l=Cape%20Town%2C%20Western%20Cape",
            PageType.LIST,
        ),
        ("https://www.linkedin.com/jobs/", PageType.LIST),
        ("https://docs.python.org/3/library/enum.html", PageType.NA),
    ]
    for url, page_type in data:
        assert derive_page_type(url) == page_type
