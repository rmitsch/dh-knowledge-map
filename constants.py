"""
Constant data
"""

API_BASE: str = "https://dhcr.clarin-dariah.eu/api/v1/"

DH_REGISTRY_MATERIALS_ENDPOINTS: dict = {
    "courses": API_BASE + "courses/index",
    "countries": API_BASE + "countries/index",
    "disciplines": API_BASE + "institutions/index",
    "universities": API_BASE + "disciplines/index"
}


