from __future__ import annotations

import os

import pandas
import pandas as pd
import pyarrow.parquet as pq

pd.options.display.max_columns = 10


# no. of triangles is 2 less than number of sides = number of vertices
def getTriangles(bounds: list[Coordinate]) -> list[Triangle]:
    triangles: list[Triangle] = []
    while len(bounds) > 2:
        triangles.append(Triangle(bounds[0], bounds[-1], bounds[-2]))
        bounds.pop()
    return triangles


class Coordinate:
    def __init__(self, x: int, y: int, z: int = 0):
        self.x: int = x
        self.y: int = y
        self.z: int = z
        self.zBounds = [285, 421]  # hard coded for now to simplify 2D comparison instead of 3D

    def zCheck(self):
        return self.zBounds[0] <= self.z <= self.zBounds[1]

    '''
    Time complexity: O( )
    '''

    def withinBoundary(self, triangles: list[Triangle]):
        for triangle in triangles:
            if triangle.containsPoint(self):
                return True
        return False

    def print(self):
        print(f"{(self.x, self.y, self.z)}")


'''
Alternative approach:
    1. Split polygon points into triangles
    2. Check if coordinate is inside any triangle
    
This assumes bounds given forms a convex polygon. Otherwise, we would have to manually split the boundary 
    points such that they form convex polygons which is still doable.
    
Time complexity: O(log(n))
    
However, this is inaccurate because of floating point rounding on machines - giving false positives or negatives 
'''


class Triangle:
    def __init__(self, v1: Coordinate, v2: Coordinate, v3: Coordinate):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def containsPoint(self, point: Coordinate):
        def sign(p1: Coordinate, p2: Coordinate, p3: Coordinate):
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

        d1 = sign(point, self.v1, self.v2)
        d2 = sign(point, self.v2, self.v3)
        d3 = sign(point, self.v3, self.v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def print(self) -> None:
        self.v1.print()
        self.v2.print()
        self.v3.print()


class ProcessGameState:
    def __init__(self, path: str):
        self.data = pd.DataFrame
        self.boundKey = 0  # key
        self.bounds = {}
        self.currBounds: list[Coordinate] = []
        self.ingest(path)
        self.cleanData()

    # clean None values
    def cleanData(self):
        self.data = self.getAlive()  # for the given tasks, only alive players are required.

    def ingest(self, path: str):
        curr_dir = os.path.dirname(__file__)
        full_path = os.path.join(curr_dir, path)
        self.data = pq.read_table(full_path).to_pandas()  # load parquet file
        self.cleanData()

    def withinBound(self, x: int, y: int, z: int, triangles: list[Triangle]):
        playerLoc = Coordinate(x, y, z)
        return playerLoc.withinBoundary(triangles) and playerLoc.zCheck()

    # write new weapon class column
    def getWeaponClasses(self, row) -> list[str]:
        weaponClasses = []
        if row is not None:
            for slot in row:
                weaponClasses.append(slot.get("weapon_class"))
        return weaponClasses

    # write list of weapon classes to new column
    def processWeaponClasses(self) -> None:
        self.data["weaponClasses"] = self.data["inventory"].map(self.getWeaponClasses)

    # write to new column "withinBounds" whether it is within the given bounds
    def processWithinBounds(self, bounds: list[Coordinate]) -> None:
        key = self.boundKey
        self.boundKey += 1
        self.bounds[key] = bounds
        triangles = getTriangles(bounds)
        self.data[f"withinBounds{key}"] = self.data.apply(
            lambda row: self.withinBound(row['x'], row['y'], row['z'], triangles), axis=1)

    def getAlive(self):
        return self.data.loc[self.data["is_alive"] == True]

    def getArea(self, area: str) -> pandas.DataFrame:
        return self.data.loc[self.data["area_name"] == area]

    def getWithinBounds(self, key: int = 0) -> pandas.DataFrame:
        return self.data.loc[self.data[f"withinBounds{key}"] == True]

    def getTeam(self, team: str) -> pandas.DataFrame:
        return self.data.loc[self.data["team"] == team]

    def getSideTeam(self, side: str, team: str) -> pandas.DataFrame:
        return self.data.loc[(self.data["side"] == side) & (self.data["team"] == team)]

    def getSideTeamWithinBounds(self, side: str, team: str, key: int = 0) -> pandas.DataFrame:
        return self.data.loc[
            (self.data["side"] == side) & (self.data["team"] == team) & (self.data[f"withinBounds{key}"] == True)]

    def getSideTeamInArea(self, side: str, team: str, area:str) -> pandas.DataFrame:
        return self.data.loc[
            (self.data["side"] == side) & (self.data["team"] == team) & (self.data["area_name"] == area)]

    def getPlayer(self, data: pandas.DataFrame, player: str) -> pandas.DataFrame:
        return data.loc[data["player"] == player]

    def getTick(self, data: pandas.DataFrame, tick: int) -> pandas.DataFrame:
        return data.loc[data["tick"] == tick]

    # viewing most relevant columns
    def view(self, data) -> pandas.DataFrame:
        return data[["tick", "side", "team", "player", "area_name", "weaponClasses", "x", "y", "z", "withinBounds0"]]
