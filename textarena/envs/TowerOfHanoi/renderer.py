from typing import List, Dict 

def create_board_str(towers: Dict[str, list]) -> str:
    """
    Render the Tower of Hanoi board as an ASCII tower.

    Args:
        towers (Dict[str, list]): Dictionary with tower names as keys and lists of disks as values.

    Returns:
        str: A pretty-printed string of the tower states.
    """
    all_disks = sum((tower for tower in towers.values()), [])
    max_disk = max(all_disks) if all_disks else 1
    height = max(len(towers[t]) for t in towers)

    tower_names = sorted(towers.keys())
    lines = []

    def render_level(tower: list, level: int) -> str:
        """Render a single level of a tower"""
        if level < len(tower):
            disk = tower[-(level + 1)]
            width = disk * 2 - 1
            return f"{'█' * width:^{max_disk * 2 + 1}}"
        else:
            return f"{'|':^{max_disk * 2 + 1}}"

    # Tower levels
    for lvl in range(height - 1, -1, -1):
        row = "   ".join(render_level(towers[t], lvl) for t in tower_names)
        lines.append(row)

    # Base and labels
    base = "   ".join("―" * (max_disk * 2 + 1) for _ in tower_names)
    labels = "   ".join(f"{t:^{max_disk * 2 + 1}}" for t in tower_names)
    lines.append(base)
    lines.append(labels)

    return "\n".join(lines)
