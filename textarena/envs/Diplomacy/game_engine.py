# good explanation of the game: https://www.youtube.com/watch?v=l53oL0ptt7k
import random
from enum import Enum
from typing import List, Optional, Dict, Set, Tuple, Any
from collections import defaultdict

from textarena.envs.Diplomacy.map_fstring import DIPLOMACY_MAP_TEMPLATE

class Season(Enum):
    SPRING = "Spring"
    FALL = "Fall"
    WINTER = "Winter"

class PhaseType(Enum):
    MOVEMENT = "Movement"
    RETREATS = "Retreats"
    ADJUSTMENTS = "Adjustments"

class UnitType(Enum):
    ARMY = "A"
    FLEET = "F"

class TerrainType(Enum):
    LAND = "land"
    SEA = "sea"
    COAST = "coast"

class OrderType(Enum):
    HOLD = "H"
    MOVE = "-"
    SUPPORT = "S"
    CONVOY = "C"
    RETREAT = "R"
    BUILD = "B"
    DISBAND = "D"
    WAIVE = "WAIVE"


class Region:
    """ Represents a region on the Diplomacy map """
    def __init__(self, name: str, terrain_type: TerrainType, is_supply_center: bool = False):
        self.name: str = name 
        self.terrain_type: TerrainType = terrain_type
        self.is_supply_center: bool = is_supply_center
        self.owner: Optional[str] = None 
        self.unit: Optional[Unit] = None 
        self.dislodged_unit: Optional[Unit] = None 
        self.adjacent_regions: Dict[str, Set[str]] = {"A": set(), "F": set()}
        self.home_for: Optional[str] = None

    def add_adjacency(self, other_region: str, unit_types: List[str]):
        """ Add adjacency to another region for specific unit types """
        for unit_type in unit_types:
            self.adjacent_regions[unit_type].add(other_region)

    def is_adjacent(self, unit_type: UnitType, other_region: str) -> bool:
        """ Check if this region is adjacent to another for a given unit type """
        return other_region in self.adjacent_regions[unit_type.value]

    def place_unit(self, unit) -> bool:
        """ Place a unit in this region """
        if self.unit is not None:
            return False 
        self.unit = unit 
        return True 

    def remove_unit(self) -> Optional['Unit']:
        """ Remove and return the unit from this region """
        unit = self.unit 
        self.unit = None 
        return unit 

    def dislodge_unit(self) -> Optional['Unit']:
        """ Remove any disloged unit """
        unit = self.dislodged_unit
        self.dislodged_unit = None 
        return unit 

    def set_owner(self, power) -> None:
        """ Set the owner of this supply center """
        if not self.is_supply_center:
            return 
        self.owner = power

    def __str__(self) -> str:
        return self.name 



class Unit:
    """ Represents a military unit on the map """
    def __init__(self, unit_type: UnitType, power: str):
        self.type: UnitType = unit_type
        self.power: str = power
        self.region: Optional[Region] = None # will be set when placed on map
        self.dislodged: bool = False 
        self.retreat_options: List[str] = [] 

    def place_in_region(self, region: Region) -> bool:
        """ Place this unit in a region """
        if region.place_unit(self):
            self.region = region 
            return True 
        return False 

    def move_to_region(self, region: Region) -> bool:
        """ Move this unit to a new region """
        if not self.region:
            return False 
        
        old_region = self.region 
        if region.place_unit(self):
            old_region.remove_unit()
            self.region = region 
            return True 
        return False 

    def dislodge(self) -> None:
        """ Mark this unit as dislodged """
        self.dislodged = True 
        if self.region:
            self.region.unit = None 

    def retreat(self, region: Region) -> bool:
        """ Retreat this unit to a new region """
        if not self.dislodged or region.name not in self.retreat_options:
            return False 

        if region.place_unit(self):
            self.dislodged = False 
            self.region = region 
            self.retreat_options = []
            return True 
        return False

    def __str__(self) -> str:
        prefix = "*" if self.dislodged else ""
        location = self.region.name if self.region else "NOWHERE"
        return f"{prefix}{self.type.value} {location}"



class Order:
    """ Represents an order in the game """
    def __init__(self, power: str, unit_type: UnitType, location: str, 
                 order_type: OrderType, target: str = None, secondary_target: str = None):
        self.power: str = power
        self.unit_type: UnitType = unit_type 
        self.location: str = location 
        self.order_type: OrderType = order_type 
        self.target: Optional[str] = target 
        self.secondary_target: Optional[str] = secondary_target
        self.result: Optional[str] = None # For storing resolution results
        self.strength: int = 1 # Base strength of the order 

    def __str__(self) -> str:
        if self.order_type == OrderType.HOLD:
            return f"{self.unit_type.value} {self.location} {self.order_type.value}"
        elif self.order_type == OrderType.MOVE:
            return f"{self.unit_type.value} {self.location} {self.order_type.value} {self.target}"
        elif self.order_type == OrderType.SUPPORT:
            if self.secondary_target:
                # Support move
                return f"{self.unit_type.value} {self.location} {self.order_type.value} {self.target} {OrderType.MOVE.value} {self.secondary_target}"
            else:
                # Support hold
                return f"{self.unit_type.value} {self.location} {self.order_type.value} {self.target}"
        elif self.order_type == OrderType.CONVOY:
            return f"{self.unit_type.value} {self.location} {self.order_type.value} {self.target} {OrderType.MOVE.value} {self.secondary_target}"
        elif self.order_type == OrderType.RETREAT:
            return f"{self.unit_type.value} {self.location} R {self.target}"
        elif self.order_type == OrderType.BUILD:
            return f"{self.unit_type.value} {self.location} B"
        elif self.order_type == OrderType.DISBAND:
            return f"{self.unit_type.value} {self.location} D"
        elif self.order_type == OrderType.WAIVE:
            return "WAIVE"
        return "Invalid Order"

    @classmethod
    def parse(cls, order_str: str, power: str) -> 'Order':
        """Parse an order string into an Order object"""
        if order_str.strip().upper() == "WAIVE":
            return cls(power, None, None, OrderType.WAIVE)
            
        parts = order_str.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Invalid order format: {order_str}")
            
        unit_type = UnitType.ARMY if parts[0] == 'A' else UnitType.FLEET
        location = parts[1]
        
        if parts[2] == 'H':
            return cls(power, unit_type, location, OrderType.HOLD)
        elif parts[2] == '-':
            if len(parts) < 4:
                raise ValueError(f"Move order missing destination: {order_str}")
            return cls(power, unit_type, location, OrderType.MOVE, parts[3])
        elif parts[2] == 'S':
            if len(parts) < 4:
                raise ValueError(f"Support order invalid: {order_str}")
            supported_unit_type = UnitType.ARMY if parts[3] == 'A' else UnitType.FLEET
            supported_location = parts[4]
            
            if len(parts) >= 7 and parts[5] == '-':
                # Support move
                return cls(power, unit_type, location, OrderType.SUPPORT, 
                          f"{supported_unit_type.value} {supported_location}", parts[6])
            else:
                # Support hold
                return cls(power, unit_type, location, OrderType.SUPPORT, 
                          f"{supported_unit_type.value} {supported_location}")
        elif parts[2] == 'C':
            if len(parts) < 7 or parts[5] != '-':
                raise ValueError(f"Convoy order invalid: {order_str}")
            convoyed_unit_type = UnitType.ARMY if parts[3] == 'A' else UnitType.FLEET
            convoyed_location = parts[4]
            return cls(power, unit_type, location, OrderType.CONVOY, 
                      f"{convoyed_unit_type.value} {convoyed_location}", parts[6])
        elif parts[2] == 'R':
            if len(parts) < 4:
                raise ValueError(f"Retreat order missing destination: {order_str}")
            return cls(power, unit_type, location, OrderType.RETREAT, parts[3])
        elif parts[2] == 'B':
            return cls(power, unit_type, location, OrderType.BUILD)
        elif parts[2] == 'D':
            return cls(power, unit_type, location, OrderType.DISBAND)
            
        raise ValueError(f"Unknown order type")



class Power:
    """ Represents a power in the game """
    def __init__(self, name: str):
        self.name: str = name 
        self.units: List[Unit] = [] 
        self.orders: List[Order] = []
        self.home_centers: List[str] = [] # List of home supply center names
        self.controlled_centers: List[str] = [] # List of currently controlled supply centers
        self.is_waiting: bool = True
        self.is_defeated: bool = False
        self.experience: List[str] = []
        self.phase_summary: str = ""
        self.plans: List[str] = []

    def __str__(self) -> str:
        return self.name
    
    def add_experience(self, experience: str) -> None:
        """ Add experience to this power """
        self.experience.append(experience)

    def add_phase_summary(self, phase_summary: str) -> None:
        """ Add phase summary to this power """
        self.phase_summary = phase_summary
    
    def add_plan(self, plan: str) -> None:
        """ Add a plan to this power """
        self.plans.append(plan)

    def add_unit(self, unit: Unit) -> None:
        """ Add a unit to this power """
        unit.power = self.name
        self.units.append(unit)

    def remove_unit(self, unit: Unit) -> None:
        """ Remove a unit from this power """
        if unit in self.units:
            self.units.remove(unit)

    def add_center(self, center: str) -> None:
        """ Add a supply center to this power """
        if center not in self.controlled_centers:
            self.controlled_centers.append(center)
    
    def remove_center(self, center: str) -> None:
        """ Remove a supply center from this power """
        if center in self.controlled_centers:
            self.controlled_centers.remove(center)

    def clear_orders(self) -> None:
        """ Clear all orders """
        self.orders = [] 

    def set_orders(self, orders: List[Order]) -> None:
        """ Set orders for this power """
        self.orders = orders 
        self.is_waiting = False 

    def check_elimination(self) -> bool:
        """ Check if this power is eliminated """
        if not self.units and not self.controlled_centers:
            self.is_defeated = True 
        return self.is_defeated

    def get_buildable_locations(self, game_map: 'Map') -> List[str]:
        """ Get locations where this power can build new units """
        buildable = []
        for center_name in self.home_centers:
            if (center_name in self.controlled_centers and 
                game_map.regions[center_name].unit is None and 
                game_map.regions[center_name].dislodged_unit is None):
                buildable.append(center_name)
        return buildable

    def count_needed_builds(self) -> int:
        """ Calculate needed builds (positive) or needed disbands (negative) """
        return len(self.controlled_centers) - len(self.units)


class Map:
    """ Represents the Diplomacy map """
    def __init__(self):
        self.regions: Dict[str, Region] = {} # Dict mapping region names to Region objects

    def add_region(self, name: str, terrain_type: TerrainType, is_supply_center: bool = False, home_for: str = None) -> None:
        """ Add a region to the map """
        region: Region = Region(name, terrain_type, is_supply_center)
        region.home_for = home_for 
        self.regions[name] = region 

    def add_adjacency(self, region1: str, region2: str, unit_types: List[str]) -> None:
        """ Add bidirectional adjacency between regions """
        if region1 in self.regions and region2 in self.regions:
            self.regions[region1].add_adjacency(region2, unit_types)
            self.regions[region2].add_adjacency(region1, unit_types)

    def get_region(self, name: str) -> Optional[Region]:
        """ Get a region by name """
        return self.regions.get(name)

    def get_all_regions(self) -> List[Region]:
        """ Get all regions on the map """
        return list(self.regions.values())

    def get_supply_centers(self) -> List[str]:
        """ Get all dupply center names """
        return [name for name, region in self.regions.items() if region.is_supply_center]

    def get_home_centers(self, power: str) -> List[str]:
        """ Get home supply centers for a power """
        return [name for name, region in self.regions.items()
                    if region.is_supply_center and region.home_for == power]

    @classmethod
    def create_standard_map(cls) -> 'Map':
        """ Create the standard Diplomacy map """
        game_map = cls()

        # Define regions
        regions_data = [
            # Format: (name, terrain_type, is_supply_center, home_power)
            # Home Supply Centers
            ('BRE', TerrainType.COAST, True, 'FRANCE'),
            ('PAR', TerrainType.LAND, True, 'FRANCE'),
            ('MAR', TerrainType.COAST, True, 'FRANCE'),
            ('LON', TerrainType.COAST, True, 'ENGLAND'),
            ('EDI', TerrainType.COAST, True, 'ENGLAND'),
            ('LVP', TerrainType.COAST, True, 'ENGLAND'),
            ('BER', TerrainType.COAST, True, 'GERMANY'),
            ('MUN', TerrainType.LAND, True, 'GERMANY'),
            ('KIE', TerrainType.COAST, True, 'GERMANY'),
            ('VEN', TerrainType.COAST, True, 'ITALY'),
            ('ROM', TerrainType.COAST, True, 'ITALY'),
            ('NAP', TerrainType.COAST, True, 'ITALY'),
            ('VIE', TerrainType.LAND, True, 'AUSTRIA'),
            ('TRI', TerrainType.COAST, True, 'AUSTRIA'),
            ('BUD', TerrainType.LAND, True, 'AUSTRIA'),
            ('CON', TerrainType.COAST, True, 'TURKEY'),
            ('ANK', TerrainType.COAST, True, 'TURKEY'),
            ('SMY', TerrainType.COAST, True, 'TURKEY'),
            ('WAR', TerrainType.LAND, True, 'RUSSIA'),
            ('MOS', TerrainType.LAND, True, 'RUSSIA'),
            ('SEV', TerrainType.COAST, True, 'RUSSIA'),
            ('STP', TerrainType.COAST, True, 'RUSSIA'),
            # Neutral Supply Centers
            ('NWY', TerrainType.COAST, True, None),
            ('SWE', TerrainType.COAST, True, None),
            ('DEN', TerrainType.COAST, True, None),
            ('HOL', TerrainType.COAST, True, None),
            ('BEL', TerrainType.COAST, True, None),
            ('SPA', TerrainType.COAST, True, None),
            ('POR', TerrainType.COAST, True, None),
            ('TUN', TerrainType.COAST, True, None),
            ('SER', TerrainType.LAND, True, None),
            ('RUM', TerrainType.COAST, True, None),
            ('BUL', TerrainType.COAST, True, None),
            ('GRE', TerrainType.COAST, True, None),
            # Non-supply centers
            ('PIC', TerrainType.COAST, False, None),
            ('BUR', TerrainType.LAND, False, None),
            ('GAS', TerrainType.COAST, False, None),
            ('YOR', TerrainType.COAST, False, None),
            ('WAL', TerrainType.COAST, False, None),
            ('CLY', TerrainType.COAST, False, None),
            ('RUH', TerrainType.LAND, False, None),
            ('PIE', TerrainType.COAST, False, None),
            ('TUS', TerrainType.COAST, False, None),
            ('APU', TerrainType.COAST, False, None),
            ('TYR', TerrainType.LAND, False, None),
            ('BOH', TerrainType.LAND, False, None),
            ('GAL', TerrainType.LAND, False, None),
            ('SIL', TerrainType.LAND, False, None),
            ('PRU', TerrainType.COAST, False, None),
            ('FIN', TerrainType.COAST, False, None),
            ('LVN', TerrainType.COAST, False, None),
            ('UKR', TerrainType.LAND, False, None),
            ('ALB', TerrainType.COAST, False, None),
            ('ARM', TerrainType.LAND, False, None),
            ('SYR', TerrainType.COAST, False, None),
            # Seas
            ('MAO', TerrainType.SEA, False, None),
            ('NAO', TerrainType.SEA, False, None),
            ('IRI', TerrainType.SEA, False, None),
            ('ENG', TerrainType.SEA, False, None),
            ('NTH', TerrainType.SEA, False, None),
            ('SKA', TerrainType.SEA, False, None),
            ('HEL', TerrainType.SEA, False, None),
            ('BAL', TerrainType.SEA, False, None),
            ('BOT', TerrainType.SEA, False, None),
            ('BAR', TerrainType.SEA, False, None),
            ('NWG', TerrainType.SEA, False, None),
            ('WES', TerrainType.SEA, False, None),
            ('LYO', TerrainType.SEA, False, None),
            ('TYS', TerrainType.SEA, False, None),
            ('ADR', TerrainType.SEA, False, None),
            ('ION', TerrainType.SEA, False, None),
            ('AEG', TerrainType.SEA, False, None),
            ('EAS', TerrainType.SEA, False, None),
            ('BLA', TerrainType.SEA, False, None),
        ]

        # Add regions to the map 
        for name, terrain, is_sc, home_for in regions_data:
            game_map.add_region(name, terrain, is_sc, home_for)

        # Add adjacencies - grouped by region for readability
        adjacencies = [
            # Western Europe
            ('BRE', 'ENG', ['F']),
            ('BRE', 'MAO', ['F']),
            ('BRE', 'PAR', ['A']),
            ('BRE', 'PIC', ['A', 'F']),
            ('BRE', 'GAS', ['A']),
            ('PAR', 'PIC', ['A']),
            ('PAR', 'BUR', ['A']),
            ('PAR', 'GAS', ['A']),
            ('MAR', 'PIE', ['A']),
            ('MAR', 'BUR', ['A']),
            ('MAR', 'GAS', ['A']),
            ('MAR', 'LYO', ['F']),
            ('MAR', 'SPA', ['A', 'F']),
            ('GAS', 'SPA', ['A']),
            ('GAS', 'BUR', ['A']),
            ('GAS', 'MAO', ['F']),
            ('PIC', 'BUR', ['A']),
            ('PIC', 'BEL', ['A', 'F']),
            ('PIC', 'ENG', ['F']),
            ('BUR', 'RUH', ['A']),
            ('BUR', 'BEL', ['A']),
            ('BUR', 'MUN', ['A']),
            
            # British Isles
            ('EDI', 'CLY', ['A', 'F']),
            ('EDI', 'YOR', ['A']),
            ('EDI', 'NTH', ['F']),
            ('EDI', 'NWG', ['F']),
            ('CLY', 'NAO', ['F']),
            ('CLY', 'NWG', ['F']),
            ('CLY', 'LVP', ['A']),
            ('LVP', 'YOR', ['A']),
            ('LVP', 'WAL', ['A']),
            ('LVP', 'IRI', ['F']),
            ('LVP', 'NAO', ['F']),
            ('YOR', 'WAL', ['A']),
            ('YOR', 'LON', ['A']),
            ('YOR', 'NTH', ['F']),
            ('WAL', 'LON', ['A']),
            ('WAL', 'ENG', ['F']),
            ('WAL', 'IRI', ['F']),
            ('LON', 'NTH', ['F']),
            ('LON', 'ENG', ['F']),
            
            # Seas around Britain
            ('IRI', 'NAO', ['F']),
            ('IRI', 'MAO', ['F']),
            ('IRI', 'ENG', ['F']),
            ('NAO', 'NWG', ['F']),
            ('NAO', 'MAO', ['F']),
            ('ENG', 'NTH', ['F']),
            ('ENG', 'MAO', ['F']),
            ('ENG', 'BEL', ['F']),
            ('NTH', 'NWG', ['F']),
            ('NTH', 'SKA', ['F']),
            ('NTH', 'DEN', ['F']),
            ('NTH', 'HEL', ['F']),
            ('NTH', 'HOL', ['F']),
            ('NTH', 'BEL', ['F']),
            ('NWG', 'BAR', ['F']),
            ('NWG', 'NWY', ['F']),
            ('BAR', 'STP', ['F']),
            ('BAR', 'NWY', ['F']),
            
            # Central Europe
            ('HOL', 'BEL', ['A', 'F']),
            ('HOL', 'KIE', ['A', 'F']),
            ('HOL', 'RUH', ['A']),
            ('HOL', 'HEL', ['F']),
            ('BEL', 'RUH', ['A']),
            ('RUH', 'KIE', ['A']),
            ('RUH', 'MUN', ['A']),
            ('KIE', 'BER', ['A', 'F']),
            ('KIE', 'MUN', ['A']),
            ('KIE', 'DEN', ['A', 'F']),
            ('KIE', 'HEL', ['F']),
            ('KIE', 'BAL', ['F']),
            ('BER', 'PRU', ['A', 'F']),
            ('BER', 'SIL', ['A']),
            ('BER', 'MUN', ['A']),
            ('BER', 'BAL', ['F']),
            ('MUN', 'BOH', ['A']),
            ('MUN', 'TYR', ['A']),
            ('MUN', 'SIL', ['A']),
            ('MUN', 'BUR', ['A']),
            
            # Scandinavia
            ('DEN', 'SKA', ['F']),
            ('DEN', 'BAL', ['F']),
            ('DEN', 'SWE', ['A', 'F']),
            ('DEN', 'HEL', ['F']),
            ('NWY', 'SWE', ['A', 'F']),
            ('NWY', 'FIN', ['A']),
            ('NWY', 'STP', ['A']),
            ('NWY', 'SKA', ['F']),
            ('SWE', 'FIN', ['A', 'F']),
            ('SWE', 'STP', ['A']),
            ('SWE', 'SKA', ['F']),
            ('SWE', 'BAL', ['F']),
            ('SWE', 'BOT', ['F']),
            ('FIN', 'STP', ['A']),
            ('FIN', 'BOT', ['F']),
            ('SKA', 'BAL', ['F']),
            
            # Baltic Region
            ('BAL', 'PRU', ['F']),
            ('BAL', 'LVN', ['F']),
            ('BAL', 'BOT', ['F']),
            ('BOT', 'STP', ['F']),
            ('BOT', 'LVN', ['F']),
            ('PRU', 'SIL', ['A']),
            ('PRU', 'WAR', ['A']),
            ('PRU', 'LVN', ['A', 'F']),
            ('SIL', 'BOH', ['A']),
            ('SIL', 'GAL', ['A']),
            ('SIL', 'WAR', ['A']),
            
            # Eastern Europe
            ('STP', 'MOS', ['A']),
            ('STP', 'LVN', ['A']),
            ('LVN', 'MOS', ['A']),
            ('LVN', 'WAR', ['A']),
            ('MOS', 'UKR', ['A']),
            ('MOS', 'SEV', ['A']),
            ('MOS', 'WAR', ['A']),
            ('WAR', 'UKR', ['A']),
            ('WAR', 'GAL', ['A']),
            ('UKR', 'SEV', ['A']),
            ('UKR', 'RUM', ['A']),
            ('UKR', 'GAL', ['A']),
            ('SEV', 'RUM', ['A', 'F']),
            ('SEV', 'ARM', ['A', 'F']),
            ('SEV', 'BLA', ['F']),
            
            # Italy and Adriatic
            ('PIE', 'TYR', ['A']),
            ('PIE', 'VEN', ['A']),
            ('PIE', 'TUS', ['A']),
            ('PIE', 'LYO', ['F']),
            ('VEN', 'TYR', ['A']),
            ('VEN', 'TRI', ['A']),
            ('VEN', 'APU', ['A']),
            ('VEN', 'ROM', ['A']),
            ('VEN', 'TUS', ['A']),
            ('VEN', 'ADR', ['F']),
            ('TYR', 'BOH', ['A']),
            ('TYR', 'VIE', ['A']),
            ('TYR', 'TRI', ['A']),
            ('TYR', 'MUN', ['A']),
            ('TUS', 'ROM', ['A']),
            ('TUS', 'LYO', ['F']),
            ('TUS', 'TYS', ['F']),
            ('ROM', 'NAP', ['A']),
            ('ROM', 'APU', ['A']),
            ('ROM', 'TYS', ['F']),
            ('NAP', 'APU', ['A']),
            ('NAP', 'ION', ['F']),
            ('NAP', 'TYS', ['F']),
            ('APU', 'ADR', ['F']),
            ('APU', 'ION', ['F']),
            
            # Mediterranean Seas
            ('LYO', 'TYS', ['F']),
            ('LYO', 'WES', ['F']),
            ('LYO', 'SPA', ['F']),
            ('WES', 'TYS', ['F']),
            ('WES', 'TUN', ['F']),
            ('WES', 'NAF', ['F']),
            ('WES', 'SPA', ['F']),
            ('TYS', 'ION', ['F']),
            ('TYS', 'TUN', ['F']),
            ('ION', 'ADR', ['F']),
            ('ION', 'ALB', ['F']),
            ('ION', 'GRE', ['F']),
            ('ION', 'TUN', ['F']),
            ('ION', 'EAS', ['F']),
            ('ION', 'AEG', ['F']),
            ('ADR', 'TRI', ['F']),
            ('ADR', 'ALB', ['F']),
            ('AEG', 'EAS', ['F']),
            ('AEG', 'GRE', ['F']),
            ('AEG', 'BUL', ['F']),
            ('AEG', 'CON', ['F']),
            ('AEG', 'SMY', ['F']),
            ('EAS', 'SMY', ['F']),
            ('EAS', 'SYR', ['F']),
            
            # Iberian Peninsula
            ('SPA', 'POR', ['A', 'F']),
            ('SPA', 'GAS', ['A']),
            ('SPA', 'MAO', ['F']),
            ('SPA', 'MAR', ['A', 'F']),
            ('POR', 'MAO', ['F']),
            ('MAO', 'NAF', ['F']),
            
            # North Africa
            ('NAF', 'TUN', ['A', 'F']),
            ('NAF', 'MAO', ['F']),
            ('TUN', 'NAF', ['A', 'F']),
            
            # Balkans
            ('BOH', 'VIE', ['A']),
            ('BOH', 'GAL', ['A']),
            ('VIE', 'GAL', ['A']),
            ('VIE', 'BUD', ['A']),
            ('VIE', 'TRI', ['A']),
            ('GAL', 'BUD', ['A']),
            ('GAL', 'RUM', ['A']),
            ('BUD', 'TRI', ['A']),
            ('BUD', 'SER', ['A']),
            ('BUD', 'RUM', ['A']),
            ('TRI', 'ALB', ['A', 'F']),
            ('TRI', 'SER', ['A']),
            ('SER', 'ALB', ['A']),
            ('SER', 'GRE', ['A']),
            ('SER', 'BUL', ['A']),
            ('SER', 'RUM', ['A']),
            ('RUM', 'BUL', ['A', 'F']),
            ('RUM', 'BLA', ['F']),
            ('BUL', 'GRE', ['A', 'F']),
            ('BUL', 'CON', ['A']),
            ('BUL', 'BLA', ['F']),
            ('BUL', 'AEG', ['F']),
            ('GRE', 'ALB', ['A', 'F']),
            ('ALB', 'ADR', ['F']),
            ('ALB', 'ION', ['F']),
            
            # Black Sea and Turkey
            ('BLA', 'ANK', ['F']),
            ('BLA', 'ARM', ['F']),
            ('BLA', 'CON', ['F']),
            ('CON', 'ANK', ['A', 'F']),
            ('CON', 'SMY', ['A']),
            ('ANK', 'ARM', ['A', 'F']),
            ('ANK', 'SMY', ['A']),
            ('ARM', 'SYR', ['A']),
            ('ARM', 'SMY', ['A']),
            ('SMY', 'SYR', ['A']),
            ('SMY', 'EAS', ['F'])
        ]
        
        for r1, r2, types in adjacencies:
            game_map.add_adjacency(r1, r2, types)

        return game_map



class DiplomacyGameEngine:
    """ The core game engine for Diplomacy """
    
    def __init__(self, rules=None, max_turns: int = 100):
        self.map: Map = Map.create_standard_map()
        self.powers: Dict[str, Power] = {} # Dict mapping power names to Power objects
        self.year: int = 1901
        self.season: Season = Season.SPRING
        self.phase: PhaseType = PhaseType.MOVEMENT 
        self.turn_number: int = 1 
        self.max_turns: int = max_turns
        self.winners: List[str] = []
        self.game_over: bool = False
        self.ascii_map_version: int = 5
        self.order_history: List[Dict[str, Any]] = [] # Track order history
        self.game_state_history: List[Dict[str, Any]] = []  # Store game state history

        # Initialize powers
        self._initialize_powers()

    def _initialize_powers(self):
        """ Initialize powers with starting units and centers """
        # Define standard powers
        standard_powers = {
            'FRANCE': [
                (UnitType.ARMY, 'PAR'),
                (UnitType.ARMY, 'MAR'),
                (UnitType.FLEET, 'BRE')
            ],
            'ENGLAND': [
                (UnitType.FLEET, 'LON'),
                (UnitType.FLEET, 'EDI'),
                (UnitType.ARMY, 'LVP')
            ],
            'GERMANY': [
                (UnitType.ARMY, 'BER'),
                (UnitType.ARMY, 'MUN'),
                (UnitType.FLEET, 'KIE')
            ],
            'ITALY': [
                (UnitType.ARMY, 'ROM'),
                (UnitType.ARMY, 'VEN'),
                (UnitType.FLEET, 'NAP')
            ],
            'AUSTRIA': [
                (UnitType.ARMY, 'VIE'),
                (UnitType.ARMY, 'BUD'),
                (UnitType.FLEET, 'TRI')
            ],
            'RUSSIA': [
                (UnitType.ARMY, 'MOS'),
                (UnitType.ARMY, 'WAR'),
                (UnitType.FLEET, 'SEV'),
                (UnitType.FLEET, 'STP')
            ],
            'TURKEY': [
                (UnitType.ARMY, 'CON'),
                (UnitType.ARMY, 'SMY'),
                (UnitType.FLEET, 'ANK')
            ]
        }

        # Create powers with their units
        for power_name, starting_units in standard_powers.items():
            power = Power(power_name)
            self.powers[power_name] = power

            # Set home centers
            power.home_centers = self.map.get_home_centers(power_name)

            # Add initial units
            for unit_type, location in starting_units:
                unit: Unit = Unit(unit_type, power_name)
                region: Optional[Region] = self.map.get_region(location)
                if region and unit.place_in_region(region):
                    power.add_unit(unit)
                
            # Set initial controlled centers
            for center in power.home_centers:
                power.add_center(center)
                region: Optional[Region] = self.map.get_region(center)
                if region:
                    region.set_owner(power_name)

    def setup_game(self, num_players) -> Dict[int, str]:
        """ Set up the game with the specified number of players """
        # assert correct player number once more
        if num_players < 3 or num_players > 7:
            raise ValueError(f"Number of players must be between 3 and 7, got {num_players}")
            

        # Select powers for the game 
        all_powers = list(self.powers.keys()) # ["AUS", "ENG", "FR", "GER", "ITA", "RUS", "TUR"]
        active_powers = random.sample(all_powers, num_players)

        # Remove unused powers
        for power_name in all_powers:
            if power_name not in active_powers:
                self.powers.pop(power_name)

        return {i: power for i, power in enumerate(active_powers)}

    def get_state(self):
        """ Get the current game state """
        units = {}
        centers = {}

        for power_name, power in self.powers.items():
            units[power_name] = [str(unit) for unit in power.units]
            centers[power_name] = power.controlled_centers.copy()

        return {
            'year': self.year,
            'season': self.season.value,
            'phase': self.phase.value,
            'turn': self.turn_number,
            'units': units,
            'centers': centers,
            'game_over': self.game_over,
            'winners': self.winners.copy() if self.winners else []
        }

    def get_orderable_locations(self, power_name: str) -> List[str]:
        """ Get locations where orders can be issued for a power """
        if power_name not in self.powers:
            return []

        power: Power = self.powers[power_name]
        orderable_locations: List[str] = []

        if self.phase == PhaseType.MOVEMENT:
            # IN movement phase, all units can be ordered
            for unit in power.units:
                if not unit.dislodged:
                    orderable_locations.append(unit.region.name)

        elif self.phase == PhaseType.RETREATS:
            # In retreat phase, only dislodged units can be ordered
            for unit in power.units:
                orderable_locations.append(unit.region.name)

        elif self.phase == PhaseType.ADJUSTMENTS:
            # Calculate build/disband count 
            build_count = power.count_needed_builds()

            if build_count > 0:
                # Can build in unoccupied home centers
                orderable_locations = power.get_buildable_locations(self.map)
            elif build_count < 0:
                # Must move units
                for unit in power.units:
                    if not unit.dislodged:
                        orderable_locations.append(unit.region.name)
        
        return orderable_locations

    def get_possible_orders(self, power_name: str) -> Dict[str, List[str]]:
        """ Get all possible orders for a power in the current phase """
        if power_name not in self.powers:
            return {}
            
        power = self.powers[power_name]
        possible_orders = {}
        
        # Get orderable locations for this power
        orderable_locations = self.get_orderable_locations(power_name)
        
        if self.phase == PhaseType.MOVEMENT:
            # For each unit, determine possible orders
            for unit in power.units:
                if unit.dislodged:
                    continue
                    
                location = unit.region.name
                if location not in orderable_locations:
                    continue
                    
                orders = []
                
                # Hold order is always possible
                orders.append(f"{unit.type.value} {location} H")
                
                # Move orders - check all adjacent regions
                for adj_region_name in unit.region.adjacent_regions[unit.type.value]:
                    orders.append(f"{unit.type.value} {location} - {adj_region_name}")
                
                # Support orders
                for adj_region_name in unit.region.adjacent_regions[unit.type.value]:
                    adj_region = self.map.get_region(adj_region_name)
                    if adj_region and adj_region.unit:
                        # Support hold
                        orders.append(f"{unit.type.value} {location} S {adj_region.unit.type.value} {adj_region_name}")
                        
                        # Support move - check where the adjacent unit can move
                        for adj_unit_dest in adj_region.adjacent_regions[adj_region.unit.type.value]:
                            # Only if the destination is also adjacent to the supporting unit
                            if adj_unit_dest in unit.region.adjacent_regions[unit.type.value]:
                                orders.append(f"{unit.type.value} {location} S {adj_region.unit.type.value} {adj_region_name} - {adj_unit_dest}")
                
                # Convoy orders (only for fleets in sea regions)
                if unit.type == UnitType.FLEET and unit.region.terrain_type == TerrainType.SEA:
                    for adj_region_name in unit.region.adjacent_regions[unit.type.value]:
                        adj_region = self.map.get_region(adj_region_name)
                        if adj_region and adj_region.unit and adj_region.unit.type == UnitType.ARMY:
                            # Find possible convoy destinations
                            for dest_region_name in self.map.regions:
                                dest_region = self.map.get_region(dest_region_name)
                                if (dest_region and 
                                    dest_region.terrain_type == TerrainType.COAST and
                                    dest_region_name != adj_region_name and
                                    self._has_possible_convoy_path(adj_region_name, dest_region_name)):
                                    orders.append(f"{unit.type.value} {location} C {adj_region.unit.type.value} {adj_region_name} - {dest_region_name}")
                
                possible_orders[location] = orders
                
        elif self.phase == PhaseType.RETREATS:
            # For each dislodged unit, determine retreat options
            for unit in power.units:
                if not unit.dislodged:
                    continue
                    
                location = unit.region.name
                orders = []
                
                # Disband is always an option
                orders.append(f"{unit.type.value} {location} D")
                
                # Retreat to valid locations
                for retreat_loc in unit.retreat_options:
                    orders.append(f"{unit.type.value} {location} R {retreat_loc}")
                
                possible_orders[location] = orders
                
        elif self.phase == PhaseType.ADJUSTMENTS:
            build_count = power.count_needed_builds()
            
            if build_count > 0:
                # Can build in unoccupied home centers
                buildable_locations = power.get_buildable_locations(self.map)
                
                for location in buildable_locations:
                    orders = []
                    region = self.map.get_region(location)
                    
                    # Can build army in any buildable location
                    orders.append(f"A {location} B")
                    
                    # Can build fleet only in coastal regions
                    if region.terrain_type == TerrainType.COAST:
                        orders.append(f"F {location} B")
                    
                    # Can also waive a build
                    orders.append("WAIVE")
                    
                    possible_orders[location] = orders
                    
            elif build_count < 0:
                # Must disband units
                for unit in power.units:
                    if not unit.dislodged:
                        location = unit.region.name
                        possible_orders[location] = [f"{unit.type.value} {location} D"]
        
        return possible_orders

    def validate_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """ Validate if an order is legal and return reason if invalid """
        if order.order_type == OrderType.WAIVE:
            # WAIVE is only valid in adjustment phase when building 
            if self.phase != PhaseType.ADJUSTMENTS:
                return False, "WAIVE orders only valid in adjustment phase"
            if order.power not in self.powers or self.powers[order.power].count_needed_builds() <= 0:
                return False, "WAIVE orders only valid when builds are available"
            return True, None

        # Get the unit that would execute this order
        unit: Optional[Unit] = self._find_unit(order.power, order.unit_type, order.location)
        if not unit:
            print(f"No {order.unit_type.value} unit found at {order.location} for {order.power}")
            return False, f"No {order.unit_type.value} unit found at {order.location} for {order.power}"

        # Validate based on order type 
        if order.order_type == OrderType.HOLD:
            # Hold is always valid for a unit
            return True, None

        elif order.order_type == OrderType.MOVE:
            # Check if destination exists
            dest_region = self.map.get_region(order.target)
            if not dest_region:
                return False, f"Destination region {order.target} does not exist"

            # Check if the move is adjacent (or can be convoyed for armies)
            if unit.region.is_adjacent(unit.type, order.target):
                return True, None

            # Check if army can be convoyed
            if unit.type == UnitType.ARMY and self._has_possible_convoy_path(unit.region.name, order.target):
                return True, None

            return False, f"Unit at {order.location} cannot move to {order.target} (not adjacent or no convoy path)"

        elif order.order_type == OrderType.SUPPORT:
            # Check if the supported unit exists
            supported_type = UnitType.ARMY if order.target.startswith("A ") else UnitType.FLEET
            supported_loc = order.target.split()[1]
            supported_unit = self._find_unit(None, supported_type, supported_loc)

            if not supported_unit:
                return False, f"No {supported_type.value} unit found at {supported_loc} to support"

            # Check if the supported location is adjacent
            if not unit.region.is_adjacent(unit.type, supported_loc):
                return False, f"Cannot support unit at {supported_loc} (not adjacent)"

            if order.secondary_target:
                # Support move - check if the destination is adjacent to the supported unit
                if not supported_unit.region.is_adjacent(supported_unit.type, order.secondary_target):
                    # Check if it could be a convoyed move
                    if (supported_unit.type == UnitType.ARMY and 
                        self._has_possible_convoy_path(supported_loc, order.secondary_target)):
                        return True, None
                    return False, f"Unit at {supported_loc} cannot move to {order.secondary_target}"

                # Check if the destination is adjacent to the supporting unit
                if not unit.region.is_adjacent(unit.type, order.secondary_target):
                    return False, f"Cannot support move to {order.secondary_target} (not adjacent to supporting unit)"

            return True, None

        elif order.order_type == OrderType.CONVOY:
            # Only fleets in water regions can convoy
            if unit.type != UnitType.FLEET:
                return False, "Only fleets can convoy"
            if unit.region.terrain_type != TerrainType.SEA:
                return False, "Convoying fleet must be in a sea region"

            # Check if the convoyed unit exists and is an army
            convoyed_type = UnitType.ARMY if order.target.startswith("A ") else UnitType.FLEET
            convoyed_loc = order.target.split()[1]
            convoyed_unit = self._find_unit(None, convoyed_type, convoyed_loc)

            if not convoyed_unit:
                return False, f"No {convoyed_type.value} unit found at {convoyed_loc} to convoy"
            if convoyed_unit.type != UnitType.ARMY:
                return False, "Only armies can be convoyed"

            # Check if the convoyed unit is adjacent to this fleet
            if not unit.region.is_adjacent(unit.type, convoyed_loc):
                return False, f"Convoying fleet not adjacent to unit at {convoyed_loc}"

            # Check if the destination is a coastal region
            dest_region = self.map.get_region(order.secondary_target)
            if not dest_region:
                return False, f"Destination region {order.secondary_target} does not exist"
            if dest_region.terrain_type != TerrainType.COAST:
                return False, f"Convoy destination {order.secondary_target} must be a coastal region"
            
            return True, None

        elif order.order_type == OrderType.RETREAT:
            # Unit must be dislodged
            if not unit.dislodged:
                return False, f"Unit at {order.location} is not dislodged and cannot retreat"

            # Check if retreat location is valid
            if order.target not in unit.retreat_options:
                return False, f"Cannot retreat to {order.target} (not a valid retreat option)"
            
            return True, None

        elif order.order_type == OrderType.BUILD:
            # Must be adjustment phase 
            if self.phase != PhaseType.ADJUSTMENTS:
                return False, "Build orders only valid in adjustment phase"

            power = self.powers[order.power]

            # Must have builds available
            if power.count_needed_builds() <= 0:
                return False, f"{order.power} has no builds available"

            # Location must be buildable home center
            buildable_locs = power.get_buildable_locations(self.map)
            if order.location not in buildable_locs:
                return False, f"Cannot build at {order.location} (not a vacant home supply center)"

            # Unit type must be valid for the terrain
            region = self.map.get_region(order.location)
            if order.unit_type == UnitType.FLEET and region.terrain_type != TerrainType.COAST:
                return False, f"Cannot build fleet at {order.location} (not a coastal region)"

            return True, None

        elif order.order_type == OrderType.DISBAND:
            # Must be adjustment phase OR retreat phase 
            if self.phase not in [PhaseType.ADJUSTMENTS, PhaseType.RETREATS]:
                return False, "Disband orders only valid in adjustment or retreat phase"

            power = self.powers[order.power]

            if self.phase == PhaseType.ADJUSTMENTS:
                # Must need to remove units
                if power.count_needed_builds() >= 0:
                    return False, f"{order.power} does not need to remove units"
            else: # RETREATS phase
                # Unit must be dislodged
                if not unit.dislodged:
                    return False, f"Unit at {order.location} is not dislodged and cannot be disbanded in retreat phase"
            
            return True, None

        return False, f"Unknown or unsupported order type: {order.order_type}"

    def _find_unit(self, power_name: Optional[str], unit_type: UnitType, location: str) -> Optional[Unit]:
        """ Find a unit by type and location, optionally filtering by power """
        region: Region = self.map.get_region(location)
        if not region:
            return None

        unit: Optional[Unit] = region.unit 
        if not unit:
            # Check if there's a dislodged unit
            unit = region.dislodged_unit

        if not unit:
            print(f"Unit {unit_type} at {location} not found")
            return None 
        
        if unit.type != unit_type:
            print(f"Unit {unit_type} at {location} is not a {unit.type}")
            return None

        if power_name and unit.power != power_name:
            print(f"Unit {unit_type} at {location} is not {power_name}")
            return None 
        
        return unit

    def _has_possible_convoy_path(self, start, end):
        """ Check if there's a possible convoy path between locations """
        # Simple BFS to find a path of fleets
        visited = set()
        queue = [(start, [])] # (location, path_so_far)

        while queue:
            current, path = queue.pop(0)

            if current == end:
                return True 

            if current in visited:
                continue 

            visited.add(current)
            current_region = self.map.get_region(current)


            # For each adjacent sea region with a fleet 
            for adj in current_region.adjacent_regions["F"]:
                adj_region = self.map.get_region(adj)
                if adj_region and adj_region.terrain_type == TerrainType.SEA:
                    # Check if there's a fleet here 
                    if adj_region.unit and adj_region.unit.type == UnitType.FLEET:
                        queue.append((adj, path + [adj]))
        
        return False 

    # TODO maybe keep track of completed and failed orders to add as observations?
    def resolve_orders(self, orders_by_power: Dict[str, List[str]]) -> Tuple[bool, Dict[str, Any]]:
        """ Process and resolve orders for all powers """
        # Reset waiting status
        for power in self.powers.values():
            power.is_waiting = True

        # Process submitted orders
        valid_orders: Dict[str, List[Order]] = {}
        invalid_orders: Dict[str, List[Dict[str, Any]]] = {}
        for power_name, orders_list in orders_by_power.items():
            invalid_orders[power_name] = []
            if power_name not in self.powers:
                invalid_orders[power_name].append({"reason": "Power not found", "orders": orders_list})
                continue

            power = self.powers[power_name]
            parsed_orders = []

            for order_str in orders_list:
                try:
                    if order_str == "```":
                        continue
                    order = Order.parse(order_str, power_name)
                    is_valid, reason = self.validate_order(order)
                    if is_valid:
                        parsed_orders.append(order)
                    else:
                        invalid_orders[power_name].append({"reason": reason, "orders": [order_str]})
                except ValueError as e:
                    # Skip unknown order types, likely model's reasoning process
                    if "Unknown order type" in str(e):
                        continue
                    invalid_orders[power_name].append({"reason": str(e), "orders": [order_str]})

            power.set_orders(parsed_orders)
            valid_orders[power_name] = parsed_orders 
        
        # Save order history
        order_record = {
            "turn": self.turn_number,
            "year": self.year,
            "season": self.season.value,
            "phase": self.phase.value,
            "valid_orders": {power: [str(order) for order in orders] for power, orders in valid_orders.items()},
            "invalid_orders": invalid_orders
        }
        self.order_history.append(order_record)

        if self.phase == PhaseType.MOVEMENT:
            self._resolve_movement(valid_orders)
        elif self.phase == PhaseType.RETREATS:
            self._resolve_retreats(valid_orders)
        elif self.phase == PhaseType.ADJUSTMENTS:
            self._resolve_adjustments(valid_orders)

        # Advance phase 
        self._advance_phase()

        # Check for game end conditinos
        self._check_victory()

        return True, self.get_state()

    def _record_game_state(self):
        """Record the current game state to history"""
        state = {
            "turn": self.turn_number,
            "season": self.season.value,
            "year": self.year,
            "phase": self.phase.value,
            "sc_counts": {power.name: len(power.controlled_centers) for power in self.powers.values()},
            "unit_counts": {power.name: len(power.units) for power in self.powers.values()},
            "territories": {
                region.name: {
                    "owner": region.unit.power if region.unit else None,
                    "unit_type": region.unit.type.value if region.unit else None,
                    "is_supply_center": region.is_supply_center,
                }
                for region in self.map.regions.values()
            }
        }
        self.game_state_history.append(state)

    def _resolve_movement(self, valid_orders: Dict[str, List[Order]]):
        """ Resolve the movement orders """
        # Maps a region to all orders targeting it
        # Maps a region to all orders targeting it
        attack_strength = {}  # {region_name: {strength: [(unit, [supporting_units])]}}
        move_targets = {}     # {region_name: [units moving there]}
        supports = {}         # {unit: [units supporting it]}
        convoys = {}          # {(start, end): [convoying fleets]}
        move_orders = {}      # {unit: target_region}
        support_orders = {}   # {unit: (target_unit, destination)}
        convoy_orders = {}    # {unit: (convoyed_unit, destination)}

        # Step 1: Identify all moves, supports, and convoys
        for power_name, orders in valid_orders.items():
            for order in orders:
                unit: Optional[Unit] = self._find_unit(power_name, order.unit_type, order.location)
                if not unit:
                    continue

                if order.order_type == OrderType.MOVE:
                    move_orders[unit] = order.target 
                    move_targets.setdefault(order.target, []).append(unit)

                elif order.order_type == OrderType.SUPPORT:
                    supported_type = UnitType.ARMY if order.target.startswith("A ") else UnitType.FLEET
                    supported_loc = order.target.split()[1]
                    supported_unit = self._find_unit(None, supported_type, supported_loc)

                    if supported_unit:
                        if order.secondary_target: # Support move
                            support_orders[unit] = (supported_unit, order.secondary_target)
                            supports.setdefault(supported_unit, []).append(unit)
                        else: # Support hold
                            support_orders[unit] = (supported_unit, None)
                            supports.setdefault(supported_unit, []).append(unit)

                elif order.order_type == OrderType.CONVOY:
                    convoyed_type = UnitType.ARMY if order.target.startswith("A ") else UnitType.FLEET
                    convoyed_loc = order.target.split()[1]
                    convoyed_unit = self._find_unit(None, convoyed_type, convoyed_loc)

                    if convoyed_unit and convoyed_unit.type == UnitType.ARMY:
                        convoy_key = (convoyed_loc, order.secondary_target)
                        convoys.setdefault(convoy_key, []).append(unit)
                        convoy_orders[unit] = (convoyed_unit, order.secondary_target)

        # Setp 2: Calculate attack strengths for all potential conflicts
        for target, attacking_units in move_targets.items():
            attack_strength[target] = {}

            # Add defending unit's strength
            defending_region = self.map.get_region(target)
            if defending_region and defending_region.unit:
                defending_unit = defending_region.unit 
                # If defender is not moving 
                if defending_unit not in move_orders:
                    defender_supports = supports.get(defending_unit, [])
                    strength = 1 + len(defender_supports)
                    attack_strength[target][strength] = [(defending_unit, defender_supports)]
            
            # Add each attacker's strength
            for attacker in attacking_units:
                attacker_supports: List[Unit] = supports.get(attacker, [])
                # Filter out invalid supports
                valid_supports: List[Unit] = []
                for support in attacker_supports:
                    # Support is valid if:
                    # 1. The supporting unit isn't dislodged
                    # 2. The supporting unit isn't being attacked from the unit it's supporting against
                    if not support.dislodged and self._is_valid_support(support, attacker, target, valid_orders):
                        valid_supports.append(support)

                strength = 1 + len(valid_supports)
                attack_strength[target].setdefault(strength, []).append((attacker, valid_supports))

        
        # Step 3: Resolve convoy disruptions
        disrupted_convoys = self._resolve_convoy_disruptions(convoys, attack_strength, supports)

        # Step 4: Resolve movements 
        self._resolve_movements(attack_strength, move_orders, disrupted_convoys)

        # Step 5: Update supply center ownership
        self._update_supply_centers()

        # Step 6: Prepare retreat options for dislodged units
        self._prepare_retreats()

    def _is_valid_support(self, supporting_unit: Unit, supported_unit: Unit, target: str, valid_orders: Dict[str, List[Order]]) -> bool:
        """ Check if a support is valid (not cut) """
        # Check if the supporting unit is being attacked
        supporting_region: Region = supporting_unit.region 

        for power_name, orders in valid_orders.items():
            for order in orders:
                if (order.order_type == OrderType.MOVE and 
                order.target == supporting_region.name and 
                power_name != supporting_unit.power):
                    # Support is cut unless the attack comes from the unit being supported
                    attacking_unit: Optional[Unit] = self._find_unit(power_name, order.unit_type, order.location)
                    if attacking_unit and attacking_unit != supported_unit:
                        return False 

        return True 

    def _resolve_convoy_disruptions(self, convoys: Dict[Tuple[str, str], List[Unit]], attacking_strength: Dict[str, Dict[int, List[Tuple[Unit, List[Unit]]]]], supports: Dict[Unit, List[Unit]]) -> Set[Tuple[str, str]]:
        """ Determine which convoys are disrupted """
        disrupted_convoys: Set[Tuple[str, str]] = set()

        # Check each convoying fleet to see if it's dislodged
        for (start, end), fleet_list in convoys.items():
            for fleet in fleet_list:
                fleet_region = fleet.region 

                # if there's an attack on this flee'ts location
                if fleet_region.name in attacking_strength:
                    strengths = sorted(attacking_strength[fleet_region.name].keys(), reverse=True)
                    if not strengths:
                        continue 

                    highest_strength = strengths[0]
                    strongest_attackers = attacking_strength[fleet_region.name][highest_strength]


                    # if the fleet is not among the strongest units at its location
                    fleet_strength = 1 
                    if fleet in supports:
                        fleet_strength += len(supports[fleet])

                    if fleet_strength < highest_strength:
                        # The convoy is disrupted 
                        disrupted_convoys.add((start, end))
                        break 
        return disrupted_convoys

    def _resolve_movements(self, attack_strength: Dict[str, Dict[int, List[Tuple[Unit, List[Unit]]]]], move_orders: Dict[Unit, str], disrupted_convoys: Set[Tuple[str, str]]) -> None:
        """ Resolve all movements based on attack strengths """
        # Track successful moves and dislodge units
        successful_moves: Dict[Unit, str] = {}
        dislodged_units: Dict[Unit, str] = {}

        # Process each location with conflicts
        for location, strength_dict in attack_strength.items():
            if not strength_dict:
                continue 

            strengths: List[int] = sorted(strength_dict.keys(), reverse=True)
            highest_strength: int = strengths[0]
            strongest_units: List[Tuple[Unit, List[Unit]]] = strength_dict[highest_strength]

            # If there's only one strongest unit or attacker
            if len(strongest_units) == 1:
                unit: Unit = strongest_units[0][0]
                supporters: List[Unit] = strongest_units[0][1]
                defending_region: Optional[Region] = self.map.get_region(location)

                # If this is an attack (not a hold)
                if unit in move_orders and move_orders[unit] == location:
                    # Check if the convoy is disrupted
                    unit_start: str = unit.region.name 
                    if (unit_start, location) in disrupted_convoys:
                        continue 

                    # Move succeeds 
                    successful_moves[unit] = location 

                    # if there's a defender, it's dislodged
                    if defending_region and defending_region.unit:
                        defender: Unit = defending_region.unit
                        # Only if defender isn't also moving
                        if defender not in move_orders:
                            dislodged_units[defender] = unit.region.name 

            # If there are multiple strongest units, everyone bounces
            else:
                # No movement occurs
                pass 

        # Execute successful moves
        for unit, destination in successful_moves.items():
            source_region: Region = unit.region 
            dest_region: Optional[Region] = self.map.get_region(destination)

            # Remove from source
            source_region.remove_unit()

            # Place in destination
            unit.region = dest_region 
            dest_region.unit = unit 

        # Process dislodgements
        for unit, attacker_loc in dislodged_units.items():
            unit.dislodge()
            unit.region.dislodged_unit = unit 

    def _update_supply_centers(self):
        """ Update supply center ownership after Fall movement """
        if self.season != Season.FALL:
            return 

        # Only update in Fall
        for region_name in self.map.get_supply_centers():
            region = self.map.get_region(region_name)
            occupying_unit: Optional[Unit] = region.unit 

            if occupying_unit:
                old_owner: Optional[str] = region.owner 
                new_owner: str = occupying_unit.power

                # Transfer ownership if changed
                if old_owner != new_owner:
                    if old_owner:
                        self.powers[old_owner].remove_center(region_name)

                    self.powers[new_owner].add_center(region_name)
                    region.set_owner(new_owner)

    def _prepare_retreats(self):
        """ Determine valid retreat locations for all dislodged units """
        # For each dislodged unit, find valid retreat locations
        for power_name, power in self.powers.items():
            for unit in power.units:
                if unit.dislodged:
                    retreat_options = [] 

                    # Check all adjacent locations
                    for adjacent in unit.region.adjacent_regions[unit.type.value]:
                        adjacent_region = self.map.get_region(adjacent)

                        # Location must be empty and not be a bounce location
                        if (adjacent_region and 
                            not adjacent_region.unit and
                            not adjacent_region.dislodged_unit):
                            retreat_options.append(adjacent)

                    unit.retreat_options = retreat_options

    def _resolve_retreats(self, valid_orders: Dict[str, List[Order]]):
        """ Resolve retreat phase orders """
        retreat_targets: Dict[str, List[Unit]] = {}  # {location: [retreating units]}
        retreat_orders: Dict[Unit, str] = {}   # {unit: destination}
        disband_units: Set[Unit] = set()

        # Collect all retreat orders
        for power_name, orders in valid_orders.items():
            for order in orders:
                unit: Optional[Unit] = self._find_unit(power_name, order.unit_type, order.location)
                if not unit or not unit.dislodged:
                    continue 

                if order.order_type == OrderType.RETREAT:
                    retreat_orders[unit] = order.target 
                    retreat_targets.setdefault(order.target, []).append(unit)
                elif order.order_type == OrderType.DISBAND:
                    disband_units.add(unit)

        # Resolve retreats - units bounce if multiple units retreat to same location
        successful_retreats = {}
        for location, units in retreat_targets.items():
            if len(units) == 1:
                successful_retreats[units[0]] = location 
            else:
                # All bounced units are disbanded
                disband_units.update(units) # TODO: check action

        # Execute successful retreats
        for unit, destination in successful_retreats.items():
            region: Optional[Region] = self.map.get_region(destination)
            unit.retreat(region)

        # Disband failed retreats
        for unit in disband_units:
            power: Power = self.powers[unit.power]
            power.remove_unit(unit)
            if unit.region:
                unit.region.dislodge_unit() # This is not a function in the Region class, does it mean dislodge = False?


    def _resolve_adjustments(self, valid_orders: Dict[str, List[Order]]):
        """ Resolve adjustment phase orders """
        for power_name, orders in valid_orders.items():
            power: Power = self.powers[power_name]
            build_count: int = power.count_needed_builds()

            # Process builds if needed
            if build_count > 0:
                builds_executed = 0 
                waives = 0

                for order in orders:
                    if order.order_type == OrderType.BUILD and builds_executed < build_count:
                        # Create and place the new unit
                        unit: Unit = Unit(order.unit_type, power_name)
                        region: Optional[Region] = self.map.get_region(order.location)

                        if unit.place_in_region(region):
                            power.add_unit(unit)
                            builds_executed += 1
                    elif order.order_type == OrderType.WAIVE:
                        waives += 1

                # Waive any remaining builds
                builds_executed += waives 

            # Process disbands if needed
            elif build_count < 0:
                disbands_needed: int = abs(build_count)
                disbands_executed: int = 0

                for order in orders:
                    if order.order_type == OrderType.DISBAND and disbands_executed < disbands_needed:
                        unit: Optional[Unit] = self._find_unit(power_name, order.unit_type, order.location)
                        if unit:
                            region: Optional[Region] = unit.region
                            power.remove_unit(unit)
                            region.remove_unit()
                            disbands_executed += 1


                # If not enough disbands were ordered, auto-disband furthest units from home
                if disbands_executed < disbands_needed:
                    units_to_disband = self._select_units_to_disband(
                        power, disbands_needed - disbands_executed
                    )

                    for unit in units_to_disband:
                        region = unit.region 
                        power.remove_unit(unit)
                        region.remove_unit()

    def _select_units_to_disband(self, power: Power, count: int) -> List[Unit]:
        """ Select units to automatically disband on distance from home centers """
        if count <= 0:
            return []


        # Calculate distance from each unit to nearest home center 
        unit_distances = []
        for unit in power.units:
            if unit.dislodged:
                continue 

            min_distance = float('inf')
            for home in power.home_centers:
                distance = self._calculate_distance(unit.region.name, home)
                min_distance = min(min_distance, distance)

            unit_distances.append((unit, min_distance))

        # Sort by distance (descending) then by unit type (fleets first)
        unit_distances.sort(key=lambda x: (-x[1], 0 if x[0].type == UnitType.FLEET else 1))
        
        return [unit for unit, _ in unit_distances[:count]]

    def _calculate_distance(self, start, end):
        """ Calculate approximate distance between two regions """
        # Simple BFS
        visited = set()
        queue = [(start, 0)] # (location, distance)

        while queue:
            current, distance = queue.pop(0)

            if current == end:
                return distance 

            if current in visited:
                continue 

            visited.add(current)
            current_region = self.map.get_region(current)

            # Add all adjacent regions 
            adjacencies = set()
            for unit_type in ["A", "F"]:
                adjacencies.update(current_region.adjacent_regions[unit_type])

            for adj in adjacencies:
                if adj not in visited:
                    queue.append((adj, distance + 1))

        return float('inf') # No path found 

    def _advance_phase(self):
        """ Advance to the next game phase """
        if self.phase == PhaseType.MOVEMENT:
            self.phase = PhaseType.RETREATS
            
        elif self.phase == PhaseType.RETREATS:
            if self.season == Season.FALL:
                self.phase = PhaseType.ADJUSTMENTS
            else:
                self.season = Season.FALL
                self.phase = PhaseType.MOVEMENT
                
        elif self.phase == PhaseType.ADJUSTMENTS:
            self.year += 1
            self.season = Season.SPRING
            self.phase = PhaseType.MOVEMENT
            self.turn_number += 1 
            
        # Reset all waiting flags
        for power in self.powers.values():
            power.is_waiting = True
            power.clear_orders()

        # After advancing phase, record the new game state
        self._record_game_state()

    def _check_victory(self):
        """ Check if any power has achieved victory """
        # Count supply centers for each power
        for power_name, power in self.powers.items():
            # Check for elimination
            power.check_elimination()

            # Check for victory
            center_count = len(power.controlled_centers)
            total_centers = len(self.map.get_supply_centers())
            victory_threshold = (total_centers // 2) + 1 

            if center_count >= victory_threshold:
                self.winners = [power_name]
                self.game_over = True 
                return 

        # Check if game should end (max turns reached or only one power remains)
        active_powers = [p for p in self.powers.values() if not p.is_defeated]
        if len(active_powers) <= 1 or self.turn_number >= self.max_turns:
            # Game ends in draw or with one winner
            if len(active_powers) == 1:
                self.winners = [active_powers[0].name]
            self.game_over = True

    def get_ascii_map(self) -> str:
        versions = {
            1: self.get_ascii_map_v1,
            2: self.get_ascii_map_v2,
            3: self.get_ascii_map_v3,
            4: self.get_ascii_map_v4,
            5: self.get_ascii_map_v5,
        }
        return versions[self.ascii_map_version]()

    def get_ascii_map_v1(self) -> str:
        """Generate a tile-based ASCII map with clear terrain and border indicators"""
        # Define power abbreviations and colors
        power_abbr = {
            'AUSTRIA': 'AUS',
            'ENGLAND': 'ENG',
            'FRANCE': 'FRA',
            'GERMANY': 'GER',
            'ITALY': 'ITA',
            'RUSSIA': 'RUS',
            'TURKEY': 'TUR',
            None: '   '
        }
        
        # Create territory ownership mapping
        territory_owners = {}
        for power_name, power in self.powers.items():
            for center in power.controlled_centers:
                territory_owners[center] = power_name
        
        # Create unit location mapping
        units_by_location = {}
        for power_name, power in self.powers.items():
            for unit in power.units:
                if unit.region:
                    units_by_location[unit.region.name] = {
                        'type': unit.type.value,
                        'power': power_name,
                        'dislodged': unit.dislodged
                    }
        
        # Create a tile for each territory with appropriate borders
        def create_territory_box(name):
            """Create a text box representing a territory"""
            if not name or name not in self.map.regions:
                return ["          ", "          ", "          ", "          ", "          ", "          "]
                
            region = self.map.regions[name]
            is_sc = region.is_supply_center
            owner = territory_owners.get(name, None)
            owner_abbr = power_abbr.get(owner, '   ')
            
            # Determine border style based on terrain
            terrain = region.terrain_type
            
            # Unit information
            unit_info = units_by_location.get(name, None)
            if unit_info:
                unit_display = f"{unit_info['type']}-{power_abbr[unit_info['power']][:3]}"
                if unit_info['dislodged']:
                    unit_display = f"*{unit_display}"
                else:
                    unit_display = f" {unit_display}"
            else:
                unit_display = "      "
            
            sc_marker = "SC" if is_sc else "  "
            
            # Create box with appropriate borders based on terrain
            if terrain == TerrainType.SEA:
                line1 = f"~~~~~~~~"
                line2 = f"~ {name:^4} ~"
                line3 = f"~      ~"
                line4 = f"~{unit_display:^6}~"
                line5 = f"~      ~"
                line6 = f"~~~~~~~~"
            elif terrain == TerrainType.COAST:
                line1 = f"+~~~~~~+"
                line2 = f"| {name:^4} |"
                line3 = f"| {sc_marker:^4} |"
                line4 = f"|{unit_display:^6}|"
                line5 = f"| {owner_abbr:^4} |"
                line6 = f"+~~~~~~+"
            else:  # LAND
                line1 = f"+------+"
                line2 = f"| {name:^4} |"
                line3 = f"| {sc_marker:^4} |"
                line4 = f"|{unit_display:^6}|"
                line5 = f"| {owner_abbr:^4} |"
                line6 = f"+------+"
            
            return [line1, line2, line3, line4, line5, line6]
        
        # Define regions by geographic area for layout
        map_layout = [
            # North
            [None, "BAR", "STP", None, None, "FIN", None],
            ["NAO", "NWG", "NTH", "SKA", "BOT", "SWE", None],
            ["IRI", "CLY", "EDI", "DEN", "BAL", "LVN", "MOS"],
            [None, "LVP", "YOR", "HEL", "BER", "PRU", "WAR"],
            
            # Central Europe
            ["MAO", "WAL", "LON", "HOL", "KIE", "SIL", "GAL", "UKR"],
            ["BRE", "ENG", "BEL", "RUH", "MUN", "BOH", "VIE", "RUM"],
            ["GAS", "PAR", "BUR", "TYR", "TRI", "BUD", "SER", "SEV"],
            ["SPA", "MAR", "PIE", "VEN", "ADR", "ALB", "BUL", "BLA"],
            
            # Mediterranean
            ["POR", "WES", "LYO", "TUS", "ROM", "ION", "GRE", "AEG"],
            ["NAF", "TYS", "APU", "NAP", "EAS", "CON", "ANK", "ARM"],
            ["TUN", None, None, None, "SYR", "SMY", None, None],
        ]
        
        # Generate the map tiles
        tile_grid = []
        for row in map_layout:
            row_tiles = []
            for region in row:
                row_tiles.append(create_territory_box(region))
            tile_grid.append(row_tiles)
        
        # Combine tiles into a map
        map_lines = []
        for row_idx, row in enumerate(tile_grid):
            # Each tile has 6 lines
            for line_idx in range(6):
                line = ""
                for tile in row:
                    if line_idx < len(tile):
                        line += tile[line_idx] + "  "
                    else:
                        line += "          "
                map_lines.append(line)
            # Add space between rows
            map_lines.append("")
        
        # Create legend with improved formatting
        legend = [
            "+----------------------------------------------------------+",
            "|                   MAP LEGEND                             |",
            "+----------------------------------------------------------+",
            "| TERRAIN TYPES:                                           |",
            "| +------+  Land region (solid border)                     |",
            "| +~~~~~~+  Coastal region (mixed border)                  |",
            "| ~~~~~~~~  Sea region (wavy border)                       |",
            "|                                                          |",
            "| TERRITORY FORMAT:                                        |",
            "| | NAME |  <- Territory name (3-letter code)              |",
            "| |  SC  |  <- Supply center (if present)                  |",
            "| |A-PWR |  <- Unit type and power (A=Army, F=Fleet)       |",
            "| | PWR  |  <- Controlling power                           |",
            "|                                                          |",
            "| UNIT FORMAT:                                             |",
            "| A-FRA = Army France                                      |",
            "| F-ENG = Fleet England                                    |",
            "| *F-TUR = Dislodged Fleet Turkey                          |",
            "+----------------------------------------------------------+",
            "|                CURRENT GAME STATE                        |",
            "+----------------------------------------------------------+",
            f"| PHASE: {self.season.value} {self.year} {self.phase.value}",
            f"| TURN: {self.turn_number}/{self.max_turns}",
            "+----------------------------------------------------------+",
            "| POWER       SUPPLY CENTERS       UNITS                   |",
        ]
        
        # Add power statistics with better spacing
        for power_name, power in self.powers.items():
            centers = len(power.controlled_centers)
            units = len(power.units)
            legend.append(f"| {power_name:<12} {centers:^18} {units:^18} |")
        
        legend.append("+----------------------------------------------------------+")
        
        # Add key connection information with improved organization
        connections_info = [
            "LAND BORDERS BETWEEN POWERS:",
            "• France/Germany: Burgundy connects Paris and Munich",
            "• Germany/Russia: Prussia connects Berlin and Warsaw",
            "• Austria/Italy: Tyrolia and Venezia connect Vienna and Rome",
            "• Austria/Russia: Galicia connects Vienna and Warsaw",
            "• Turkey/Russia: Armenia connects Constantinople and Sevastopol",
            "",
            "SEA BORDERS:",
            "• England/France: English Channel connects London and Brest",
            "• Italy/Austria: Adriatic Sea connects Venice and Trieste",
            "• Turkey/Russia: Black Sea connects Constantinople and Sevastopol",
            "",
            "STRATEGIC STRAITS:",
            "• Denmark connects North Sea and Baltic Sea",
            "• Constantinople connects Black Sea and Aegean Sea",
            "• Gibraltar (between Spain and North Africa) connects Atlantic and Mediterranean",
            "",
            "COASTAL REGIONS WITH MULTIPLE COASTS:",
            "• Spain has North (Atlantic) and South (Mediterranean) coasts",
            "• St. Petersburg has North (Barents Sea) and South (Gulf of Bothnia) coasts",
            "• Bulgaria has East (Black Sea) and South (Aegean Sea) coasts"
        ]
        
        # Combine everything
        return "\n".join(map_lines) + "\n\n" + "\n".join(legend) + "\n\n" + "\n".join(connections_info)

    def get_ascii_map_v2(self) -> str:
        """
        Generate a classic-style ASCII map using a single pre-drawn template.
        Territory names, supply centers, and unit info are overlaid at
        specific coordinates.
        """
        # -------------------------------------------------------------
        # 1) The big ASCII map template: paste the 1989 text here
        # -------------------------------------------------------------
        ASCII_MAP_TEMPLATE = r"""
+-------------+-------------------------+-----------------------------------+
|.............|.........................|...................................|
|.........+---+...........+-------------+...............BAR.................|
|.........|   |...........|             |...................................|
|...NAO...| C |...........|             +-----------------------------------+
|.........| L |...........|             |                      nc           |
|.....+---+ Y |....NWG....|             +-----------------+                 |
|.....|   |   |...........|    =NWY=    |       FIN       |                 |
|.....|   +---+...........|             +---------+-------+                 |
|.....|   |   |...........|             |         |.......|                 |
|.....| L | E |...........|             +-----+   |.......|                 |
|.....|=V=|=D=|...........|             |.....|   |.......|      =STP=      |
|.....| P | I +-----------+-------------+.SKA.|   |.......|                 |
|.....|   |   |.........................|.....| S |.......|                 |
|.+---+   +---+..........NTH............+-----+=W=|..GOB..|sc               |
|.|...|   | Y |.........................|     | E |.......|                 |
|.|...+---+ O |.....+---+-------+-------+     |   |.......|                 |
|.|...|   | R |.....|   | =HOL= |..HEL..|=DEN=|   |.......|                 |
|.|...| W +---+.....|   | +-----+-------+     |   |.......|                 |
|.|...| A | L |.....|   | |             |     |   |.......|                 |
|.|...| L |=O=|.....|   | |             +-----+---+---+---+-----------+     |
|.|.I.|   | N |.....|   | |    =KIE=    |.....BAL.....|      LVN      |     |
|.|.R.+---+---+.....|   | |             +-----+-------+---+---+-------+-----+
|.|.I.|.......|.....|   | |             |     |           |   |             |
|.|...|.......+-----+   +-+-+-----------+=BER=|    PRU    | W |    =MOS=    |
|.|...|.......|         |   |           |     |           |=A=|             |
|.|...|..ENG..|   =BEL= | R |           +-----+-----------+ R +-+-----------+
|.|...|.......|         | U |  =MUN=    |       SIL       |   | |           |
|.|...|.......+-------+ | H |           +-------------+---+---+ |           |
|.|...|.......|  PIC  | |   |           |             |       |U|           |
+-+---+---+---+-+---+-+-+---+-+-+-------+     BOH     |       |K|   =SEV=   |
|.........|     |   |         | |       |             |  GAL  |R|           |
|.........|     | P |         | |       +-------------+       | |           |
|.........|=BRE=|=A=|  BUR    | |  TYR  |    =VIE=    |       | |           |
|.........|     | R |       +-+ |       +-----------+-+---+---+-+-+-------+-+
|.........|     |   |       |SWZ|       |           |     |       |.......| |
|.........+-----+---+-+-----+-+-+-+-----+    =TRI=  |=BUD=|       |.......| |
|.........|           |       |   |     |           |     | =RUM= |.......| |
|.........|           |       | P |     +-------+-+-+-----+       |..BLA..| |
|.........|    GAS    | =MAR= | I |=VEN=|...ADR.| |       |       |.......| |
|...MAO...|           |       | E |     +-----+.| | =SER= +-------+.......|A|
|.........|           |       |   |     |     |.| |       |     ec|.......|R|
|.........+-----------+---+---+-+-+-+---+ APU |.|A+-------+ =BUL= +---+---+M|
|.........|nc             |.....| T | R |     |.|L|       |   sc  | C | A | |
|.........+-----+         |.....| U |=O=+---+ |.|B|       +-------+=O=|=N=| |
|.........|=POR=|   =SPA= |.GOL.| S | M | N | |.| | =GRE= |.......| N | K | |
|.........+-----+         |.....+-+-+---+=A=| |.| |       |.......+---+---+ |
|.........|sc           sc|.......|.....| P | |.| |       |..AEG..|       | |
|.........+---------------+-------+.TYS.+---+-+-+-+-------+.......| =SMY= +-+
|.........|...........WMD.........|.....|.................|.......|       |S|
|.........+-----------------+-----+-----+.......ION.......+-------+-------+Y|
|.........|         NAF     |  =TUN=    |.................|......EMD......|R|
+---------+-----------------+-----------+-----------------+---------------+-+
    """.strip('\n')

        # -------------------------------------------------------------
        # 2) Define row/column positions for each region label
        #    (You must figure these out by trial/error or counting lines)
        # -------------------------------------------------------------
        region_positions = {
            "STP": (2, 20),  # Example: row=2, col=20
            "NWG": (3, 5),
            "FIN": (2, 40),
            "SWE": (3, 50),
            # ... add all other territories
        }

        # -------------------------------------------------------------
        # 3) Create a mutable structure from the template
        #    (so we can overwrite certain positions)
        # -------------------------------------------------------------
        map_lines = [list(line) for line in ASCII_MAP_TEMPLATE.split('\n')]


        # -------------------------------------------------------------
        # 4) Prepare data: territory owners, units, etc.
        # -------------------------------------------------------------
        power_abbr = {
            'AUSTRIA': 'AUS',
            'ENGLAND': 'ENG',
            'FRANCE':  'FRA',
            'GERMANY': 'GER',
            'ITALY':   'ITA',
            'RUSSIA':  'RUS',
            'TURKEY':  'TUR',
            None:      '   '
        }

        # Who owns each supply center?
        territory_owners = {}
        for power_name, power in self.powers.items():
            for center in power.controlled_centers:
                territory_owners[center] = power_name

        # Which units are in which regions?
        units_by_location = {}
        for power_name, power in self.powers.items():
            for unit in power.units:
                if unit.region:
                    units_by_location[unit.region.name] = {
                        'type': unit.type.value,    # 'A' or 'F'
                        'power': power_name,
                        'dislodged': unit.dislodged
                    }

        # -------------------------------------------------------------
        # 5) Overlay territory labels, supply centers, and units
        # -------------------------------------------------------------
        def place_text(r, c, text):
            """Helper to place text at map_lines[r][c..] (if in range)."""
            if 0 <= r < len(map_lines):
                line = map_lines[r]
                for i, ch in enumerate(text):
                    if 0 <= c + i < len(line):
                        line[c + i] = ch

        for region_name, (row, col) in region_positions.items():
            region_label = region_name.upper()[:3]

            # (a) Place the region label
            place_text(row, col, region_label)

            # (b) If it's a supply center, add a marker after the label
            region_obj = self.map.regions.get(region_name)
            if region_obj and region_obj.is_supply_center:
                place_text(row, col + len(region_label), "*")

            # (c) If there's a unit, place it on the next line (row+1)
            unit_info = units_by_location.get(region_name)
            if unit_info:
                # Example format: "A-FRA" or "*F-TUR" if dislodged
                unit_str = f"{unit_info['type']}-{power_abbr[unit_info['power']]}"
                if unit_info['dislodged']:
                    unit_str = "*" + unit_str
                place_text(row + 1, col, unit_str)

        # -------------------------------------------------------------
        # 6) Convert the map back to a single string
        # -------------------------------------------------------------
        final_map_lines = ["".join(chars) for chars in map_lines]
        final_map = "\n".join(final_map_lines)

        # -------------------------------------------------------------
        # 7) (Optional) Add your legend and other info
        # -------------------------------------------------------------
        legend = [
            " +------------------------------------------------------+",
            " |                    MAP LEGEND                         |",
            " +------------------------------------------------------+",
            " |  * indicates a supply center.                         |",
            " |  A-FRA means an Army from France, etc.                |",
            " |  *A-GER means a dislodged Army from Germany.          |",
            " +------------------------------------------------------+",
            f" | PHASE: {self.season.value} {self.year} {self.phase.value}",
            f" | TURN: {self.turn_number}/{self.max_turns}",
            " +------------------------------------------------------+",
            " | POWER       SUPPLY CENTERS       UNITS               |",
        ]

        # Example: show each power's centers/units
        for power_name, power in self.powers.items():
            centers_count = len(power.controlled_centers)
            units_count   = len(power.units)
            legend.append(f" | {power_name:<12} {centers_count:^18} {units_count:^18} |")

        legend.append(" +------------------------------------------------------+")

        # -------------------------------------------------------------

        # 8) Return the combined map + legend
        # -------------------------------------------------------------
        return final_map + "\n\n" + "\n".join(legend) + "\n"
    
    def get_ascii_map_v3(self) -> str:
        """Generate a tile-based ASCII art representation of the current game state with improved spacing"""
        # Define power abbreviations
        power_abbr = {
            'AUSTRIA': 'AUS',
            'ENGLAND': 'ENG',
            'FRANCE': 'FRA',
            'GERMANY': 'GER',
            'ITALY': 'ITA',
            'RUSSIA': 'RUS',
            'TURKEY': 'TUR',
            None: '   '
        }
        
        # Create territory ownership mapping
        territory_owners = {}
        for power_name, power in self.powers.items():
            for center in power.controlled_centers:
                territory_owners[center] = power_name
        
        # Create unit location mapping
        units_by_location = {}
        for power_name, power in self.powers.items():
            for unit in power.units:
                if unit.region:
                    units_by_location[unit.region.name] = {
                        'type': unit.type.value,
                        'power': power_name,
                        'dislodged': unit.dislodged
                    }
        
        # Create a tile for each territory
        def create_territory_box(name):
            """Create a text box representing a territory"""
            if not name or name not in self.map.regions:
                return ["          ", "          ", "          ", "          ", "          ", "          ", "          "]
                
            region = self.map.regions[name]
            is_sc = region.is_supply_center
            owner = territory_owners.get(name, None)
            owner_abbr = power_abbr.get(owner, '   ')
            
            # Unit information
            unit_info = units_by_location.get(name, None)
            if unit_info:
                unit_display = f"{unit_info['type']}-{power_abbr[unit_info['power']]}"
                if unit_info['dislodged']:
                    unit_display = f"*{unit_display}"
                else:
                    unit_display = f" {unit_display}"
            else:
                unit_display = "     "
            
            # Create box
            line1 = f"+--------+"
            line2 = f"|  {name:^4}  |"
            line3 = f"|        |"
            line4 = f"|{('SC' if is_sc else '  '):^8}|"
            line5 = f"|{unit_display:^8}|"
            line6 = f"|{owner_abbr:^8}|"
            line7 = f"+--------+"
            
            return [line1, line2, line3, line4, line5, line6, line7]
        
        # Define regions by geographic area for layout
        map_layout = [
            # North
            [None, "BAR", "STP", None, None, "FIN", None],
            ["NAO", "NWG", "NTH", "SKA", "BOT", "SWE", None],
            ["IRI", "CLY", "EDI", "DEN", "BAL", "LVN", "MOS"],
            [None, "LVP", "YOR", "HEL", "BER", "PRU", "WAR"],
            
            # Central Europe
            ["MAO", "WAL", "LON", "HOL", "KIE", "SIL", "GAL", "UKR"],
            ["BRE", "ENG", "BEL", "RUH", "MUN", "BOH", "VIE", "RUM"],
            ["GAS", "PAR", "BUR", "TYR", "TRI", "BUD", "SER", "SEV"],
            ["SPA", "MAR", "PIE", "VEN", "ADR", "ALB", "BUL", "BLA"],
            
            # Mediterranean
            ["POR", "WES", "LYO", "TUS", "ROM", "ION", "GRE", "AEG"],
            ["NAF", "TYS", "APU", "NAP", "EAS", "CON", "ANK", "ARM"],
            ["TUN", None, None, None, "SYR", "SMY", None, None],
        ]
        
        # Generate the map tiles
        tile_grid = []
        for row in map_layout:
            row_tiles = []
            for region in row:
                row_tiles.append(create_territory_box(region))
            tile_grid.append(row_tiles)
        
        # Combine tiles into a map
        map_lines = []
        for row_idx, row in enumerate(tile_grid):
            # Each tile has 7 lines
            for line_idx in range(7):
                line = ""
                for tile in row:
                    if line_idx < len(tile):
                        line += tile[line_idx] + "  "
                    else:
                        line += "            "
                map_lines.append(line)
            # Add space between rows
            map_lines.append("")
        
        # Create legend with improved formatting
        legend = [
            "+-------------------------------------------------------+",
            "|                     TERRITORY FORMAT                  |",
            "+-------------------------------------------------------+",
            "| +--------+                                           |",
            "| |  NAME  |  <- Territory name (3-letter code)        |",
            "| |        |                                           |",
            "| |   SC   |  <- Supply center (if present)            |",
            "| |  X-YYY |  <- Unit type and power (X=A/F, YYY=power)|",
            "| |  ZZZ   |  <- Controlling power (ZZZ=power)         |",
            "| +--------+                                           |",
            "|                                                       |",
            "| Unit Format:   A-ENG = Army England                  |",
            "|                F-RUS = Fleet Russia                  |",
            "|                *F-TUR = Dislodged Fleet Turkey       |",
            "+-------------------------------------------------------+",
            "|           CURRENT GAME STATE                          |",
            "+-------------------------------------------------------+",
            f"| PHASE: {self.season.value} {self.year} {self.phase.value}",
            f"| TURN: {self.turn_number}/{self.max_turns}",
            "+-------------------------------------------------------+",
            "| POWER       SUPPLY CENTERS       UNITS                |",
        ]
        
        # Add power statistics with better spacing
        for power_name, power in self.powers.items():
            centers = len(power.controlled_centers)
            units = len(power.units)
            legend.append(f"| {power_name:<12} {centers:^18} {units:^18} |")
        
        legend.append("+-------------------------------------------------------+")
        
        # Add key connection information
        connections = [
            "KEY STRATEGIC REGIONS AND CONNECTIONS:",
            "",
            "STRAITS:",
            "• Denmark (DEN) - Connects North Sea (NTH) to Baltic Sea (BAL)",
            "• Constantinople (CON) - Connects Black Sea (BLA) to Aegean Sea (AEG)",
            "• Bulgaria (BUL) - Has separate coasts on Black Sea and Aegean Sea",
            "",
            "STRATEGIC WATERWAYS:",
            "• English Channel (ENG) - Key waterway between Britain and mainland Europe",
            "• Mid-Atlantic Ocean (MAO) - Critical passage to Western Mediterranean",
            "• Tyrrhenian Sea (TYS) - Controls access to Italian supply centers",
            "• Ionian Sea (ION) - Strategic Mediterranean crossroads",
            "",
            "CRITICAL LAND ROUTES:",
            "• Burgundy (BUR) - Controls movement through Western Europe",
            "• Tyrolia (TYR) - Mountain pass connecting Germany, Austria and Italy",
            "• Galicia (GAL) - Key buffer zone between Russia, Austria and Germany",
            "• Ukraine (UKR) - Controls Eastern European approaches",
            "",
            "DEFENSIBLE POSITIONS:",
            "• Munich (MUN) - Protects Southern Germany",
            "• Vienna (VIE) - Central to Austrian defense",
            "• Moscow (MOS) - Russian heartland",
            "• Constantinople (CON) - Turkish stronghold between Europe and Asia"
        ]
        
        # Combine everything
        return "\n".join(map_lines) + "\n\n" + "\n".join(legend) + "\n\n" + "\n".join(connections)            

    def get_ascii_map_v4(self) -> str:
        """Generate a tile-based ASCII map with territory connections"""
        # Define power abbreviations
        power_abbr = {
            'AUSTRIA': 'AUS',
            'ENGLAND': 'ENG',
            'FRANCE': 'FRA',
            'GERMANY': 'GER',
            'ITALY': 'ITA',
            'RUSSIA': 'RUS',
            'TURKEY': 'TUR',
            None: '   '
        }
        
        # Create territory ownership mapping
        territory_owners = {}
        for power_name, power in self.powers.items():
            for center in power.controlled_centers:
                territory_owners[center] = power_name
        
        # Create unit location mapping
        units_by_location = {}
        for power_name, power in self.powers.items():
            for unit in power.units:
                if unit.region:
                    units_by_location[unit.region.name] = {
                        'type': unit.type.value,
                        'power': power_name,
                        'dislodged': unit.dislodged
                    }
        
        # Create a tile for each territory
        def create_territory_box(name):
            """Create a text box representing a territory"""
            if not name or name not in self.map.regions:
                return ["     ", "     ", "     ", "     "]
                
            region = self.map.regions[name]
            is_sc = region.is_supply_center
            owner = territory_owners.get(name, None)
            owner_abbr = power_abbr.get(owner, '   ')
            
            # Unit information
            unit_info = units_by_location.get(name, None)
            if unit_info:
                unit_display = f"{unit_info['type']}-{power_abbr[unit_info['power']][:1]}"
                if unit_info['dislodged']:
                    unit_display = f"*{unit_display}"
                else:
                    unit_display = f" {unit_display}"
            else:
                unit_display = "    "
            
            # Create box
            line1 = f"+-----+"
            line2 = f"|{name:^5}|" 
            line3 = f"|{('SC' if is_sc else '  '):^5}|"
            line4 = f"|{unit_display:^5}|"
            line5 = f"|{owner_abbr:^5}|"
            line6 = f"+-----+"
            
            return [line1, line2, line3, line4, line5, line6]
        
        # Define regions by geographic area for layout
        map_layout = [
            # North
            [None, "BAR", "STP", None, None],
            ["NAO", "NWG", "NTH", "SKA", "BOT"],
            ["IRI", "CLY", "EDI", "DEN", "BAL"],
            [None, "LVP", "YOR", "HEL", "BER"],
            
            # Central Europe
            ["MAO", "WAL", "LON", "HOL", "KIE"],
            ["BRE", "ENG", "BEL", "RUH", "MUN"],
            ["GAS", "PAR", "BUR", "BOH", "VIE"],
            ["SPA", "MAR", "PIE", "TYR", "TRI"],
            
            # Eastern Europe
            [None, "FIN", "SWE", "LVN", "MOS"],
            [None, None, "PRU", "WAR", "UKR"],
            [None, None, "SIL", "GAL", "RUM"],
            [None, None, "SER", "BUD", "SEV"],
            
            # Mediterranean
            ["POR", "WES", "LYO", "VEN", "ADR"],
            ["NAF", "TYS", "TUS", "ROM", "ALB"],
            ["TUN", "ION", "NAP", "GRE", "AEG"],
            [None, "EAS", None, "BUL", "BLA"],
            
            # Near East
            [None, None, "SYR", "CON", "ANK"],
            [None, None, None, None, "SMY"]
        ]
        
        # Generate the map tiles
        tile_grid = []
        for row in map_layout:
            row_tiles = []
            for region in row:
                row_tiles.append(create_territory_box(region))
            tile_grid.append(row_tiles)
        
        # Combine tiles into a map
        map_lines = []
        for row_idx, row in enumerate(tile_grid):
            # Each tile has 6 lines
            for line_idx in range(6):
                line = ""
                for tile in row:
                    if line_idx < len(tile):
                        line += tile[line_idx]
                    else:
                        line += "      "
                map_lines.append(line)
        
        # Add connections between key territories
        # (This could be expanded with actual connections from the adjacency data)
        connections = [
            "KEY CONNECTIONS:",
            "Water regions connect to adjacent coastal territories",
            "Land regions connect to adjacent territories",
            "",
            "IMPORTANT STRAITS:",
            "- Denmark (DEN): Connects NTH to BAL",
            "- Constantinople (CON): Connects BLA to AEG",
            "- Bulgaria (BUL): Has separate coasts on BLA and AEG",
            "- Spain (SPA): Has separate coasts on MAO and WES",
            "- St. Petersburg (STP): Has separate coasts on BAR and BOT"
        ]
        
        # Create legend
        legend = [
            "+---------------------------------------+",
            "| TERRITORY FORMAT                      |",
            "| +-----+                              |",
            "| |NAME |  <- Territory name           |",
            "| | SC  |  <- Supply center (if shown) |",
            "| | A-P |  <- Unit type and power      |",
            "| |OWNER|  <- Controlling power        |",
            "| +-----+                              |",
            "|                                      |",
            "| Unit Format: A-P = Army/Fleet-Power  |",
            "| * = Dislodged unit                   |",
            "+---------------------------------------+",
            "| POWER       SUPPLY CENTERS    UNITS   |"
        ]
        
        # Add power statistics
        for power_name, power in self.powers.items():
            centers = len(power.controlled_centers)
            units = len(power.units)
            legend.append(f"| {power_name:<10} {centers:^15} {units:^7} |")
        
        legend.append("+---------------------------------------+")
        
        # Add game status
        status = [
            f"GAME STATUS: {self.season.value} {self.year} {self.phase.value}",
            f"TURN: {self.turn_number}/{self.max_turns}"
        ]
        
        # Combine everything
        return "\n".join(map_lines) + "\n\n" + "\n".join(legend) + "\n\n" + "\n".join(connections) + "\n\n" + "\n".join(status)

    def get_ascii_map_v5(self) -> str:
        """Render the current game state as an ASCII map"""
        values = {}
        
        def initialize_empty_values():
            # Get all placeholder patterns from map_fstring
            import re
            placeholders = re.findall(r'{([^}]+)}', DIPLOMACY_MAP_TEMPLATE)
            for p in placeholders:
                values[p] = "         "  # 9 spaces

        def pad_center(text: str, width: int = 9) -> str:
            text = text[:width]  # Truncate if too long
            padding = width - len(text)
            left_pad = padding // 2
            right_pad = padding - left_pad
            return " " * left_pad + text + " " * right_pad

        def format_region(region_code: str, region: Region) -> None:
            """Format a single region's display values"""
            # Row 1: Region abbreviation (centered)
            values[f"{region_code}_nam"] = pad_center(region_code.upper())
            
            # Row 2: Supply center info
            sc_info = ""
            if region.is_supply_center:
                if region.owner:
                    sc_info = f"SC {region.owner[:3]}"
                else:
                    sc_info = "SC"
            values[f"{region_code}_sc_"] = pad_center(sc_info)
            
            # Row 3: Unit info
            unit_info = ""
            if region.unit:
                prefix = "*" if region.unit.dislodged else ""
                unit_info = f"{prefix}{region.unit.type.value} {region.unit.power[:3]}"
            values[f"{region_code}_uni"] = pad_center(unit_info)

        # Initialize empty values
        initialize_empty_values()
        
        # Process regions
        for region_code, region in self.map.regions.items():
            region_code = region_code.lower()
            format_region(region_code, region)
        
        # Add power summary values with consistent padding
        power_codes = ['fra', 'eng', 'ger', 'aus', 'rus', 'tur']
        for power_code in power_codes:
            # Default to padded spaces
            values[f"{power_code}_scs"] = " " * 9  # 9 spaces for supply centers
            values[f"{power_code}_uns"] = " " * 9  # 9 spaces for units
            
            # If power exists, update with actual values
            power_name = {'fra': 'FRANCE', 'eng': 'ENGLAND', 'ger': 'GERMANY',
                        'aus': 'AUSTRIA', 'rus': 'RUSSIA', 'tur': 'TURKEY'}[power_code].upper()
            
            for power in self.powers.values():
                if power.name == power_name:
                    sc_count = len(power.controlled_centers)
                    unit_count = len(power.units)
                    values[f"{power_code}_scs"] = str(sc_count).center(9)
                    values[f"{power_code}_uns"] = str(unit_count).center(9)
                    break
        
        # Apply the values to the template
        return DIPLOMACY_MAP_TEMPLATE.format(**values)


    def get_ascii_map_v5(self) -> str:
        """Render the current game state as an ASCII map"""
        values = {}
        
        def initialize_empty_values():
            # Get all placeholder patterns from map_fstring
            import re
            placeholders = re.findall(r'{([^}]+)}', DIPLOMACY_MAP_TEMPLATE)
            for p in placeholders:
                values[p] = "         "  # 9 spaces

        def pad_center(text: str, width: int = 9) -> str:
            text = text[:width]  # Truncate if too long
            padding = width - len(text)
            left_pad = padding // 2
            right_pad = padding - left_pad
            return " " * left_pad + text + " " * right_pad

        def format_region(region_code: str, region: Region) -> None:
            """Format a single region's display values"""
            # Row 1: Region abbreviation (centered)
            values[f"{region_code}_nam"] = pad_center(region_code.upper())
            
            # Row 2: Supply center info
            sc_info = ""
            if region.is_supply_center:
                if region.owner:
                    sc_info = f"SC {region.owner[:3]}"
                else:
                    sc_info = "SC"
            values[f"{region_code}_sc_"] = pad_center(sc_info)
            
            # Row 3: Unit info
            unit_info = ""
            if region.unit:
                prefix = "*" if region.unit.dislodged else ""
                unit_info = f"{prefix}{region.unit.type.value} {region.unit.power[:3]}"
            values[f"{region_code}_uni"] = pad_center(unit_info)

        # Initialize empty values
        initialize_empty_values()
        
        # Process regions
        for region_code, region in self.map.regions.items():
            region_code = region_code.lower()
            format_region(region_code, region)
        
        # Add power summary values with consistent padding
        power_codes = ['fra', 'eng', 'ger', 'aus', 'rus', 'tur']
        for power_code in power_codes:
            # Default to padded spaces
            values[f"{power_code}_scs"] = " " * 9  # 9 spaces for supply centers
            values[f"{power_code}_uns"] = " " * 9  # 9 spaces for units
            
            # If power exists, update with actual values
            power_name = {'fra': 'FRANCE', 'eng': 'ENGLAND', 'ger': 'GERMANY',
                        'aus': 'AUSTRIA', 'rus': 'RUSSIA', 'tur': 'TURKEY'}[power_code].upper()
            
            for power in self.powers.values():
                if power.name == power_name:
                    sc_count = len(power.controlled_centers)
                    unit_count = len(power.units)
                    values[f"{power_code}_scs"] = str(sc_count).center(9)
                    values[f"{power_code}_uns"] = str(unit_count).center(9)
                    break
        
        # Apply the values to the template
        return DIPLOMACY_MAP_TEMPLATE.format(**values)

    def bfs_shortest_path(self, start, match_func, allowed_unit_types=None):
        """
        Find shortest path from start to a territory matching match_func.
        
        Args:
            start: Starting territory
            match_func: Function that returns the territory if it matches, None otherwise
            allowed_unit_types: Set of unit types ('A', 'F') allowed for movement
        
        Returns:
            (path, matched_territory) or (None, None) if no path found
        """
        if match_func(start):
            return [start], start
            
        visited = set([start])
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            # Check adjacent territories
            if current in self.map.regions:
                for adj_name in self.map.regions[current].adjacencies:
                    # Skip if already visited
                    if adj_name in visited:
                        continue
                        
                    # Check if unit type can move along this edge
                    if allowed_unit_types:
                        can_move = False
                        for unit_type in allowed_unit_types:
                            if unit_type in self.map.regions[current].adjacencies[adj_name]:
                                can_move = True
                                break
                        if not can_move:
                            continue
                    
                    # Check if this territory matches
                    matched = match_func(adj_name)
                    if matched:
                        return path + [adj_name], matched
                    
                    # Add to queue for further exploration
                    visited.add(adj_name)
                    queue.append((adj_name, path + [adj_name]))
                    
        return None, None
        
    def bfs_nearest_adjacent(self, start, occupant_map, allowed_unit_types=None):
        """
        Find shortest path from start to a territory adjacent to one in occupant_map.
        
        Args:
            start: Starting territory
            occupant_map: Dict mapping territory -> unit
            allowed_unit_types: Set of unit types ('A', 'F') allowed for movement
        
        Returns:
            (path, (matched_territory, matched_unit)) or (None, (None, None)) if no path found
        """
        # Check if start is already adjacent to an occupied territory
        if start in self.map.regions:
            for adj_name in self.map.regions[start].adjacencies:
                if adj_name in occupant_map:
                    return [start], (adj_name, occupant_map[adj_name])
        
        visited = set([start])
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            # Check adjacent territories
            if current in self.map.regions:
                for adj_name in self.map.regions[current].adjacencies:
                    # Skip if already visited
                    if adj_name in visited:
                        continue
                        
                    # Check if unit type can move along this edge
                    if allowed_unit_types:
                        can_move = False
                        for unit_type in allowed_unit_types:
                            if unit_type in self.map.regions[current].adjacencies[adj_name]:
                                can_move = True
                                break
                        if not can_move:
                            continue
                    
                    # Check if this territory is adjacent to an occupied territory
                    if adj_name in self.map.regions:
                        for adj_adj_name in self.map.regions[adj_name].adjacencies:
                            if adj_adj_name in occupant_map:
                                return path + [adj_name], (adj_adj_name, occupant_map[adj_adj_name])
                    
                    # Add to queue for further exploration
                    visited.add(adj_name)
                    queue.append((adj_name, path + [adj_name]))
                    
        return None, (None, None)
