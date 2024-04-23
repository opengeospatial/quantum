# -*- coding: utf-8 -*-

import json
import time
from uuid import UUID

import networkx as nx
import dwave_networkx as dnx
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler

import arcpy
from arcgis.gis import GIS
from arcgis.graph import KnowledgeGraph


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Quantum Solver"
        self.alias = "QuantumSolver"
        self.tools = [TravellingSalesperson, StructuralImbalance]


class TravellingSalesperson(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Quantum TSP Solver"
        self.description = "Solve the TSP using a D-Wave quantum computer"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""

        param0 = arcpy.Parameter(
            displayName="Input Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        param1 = arcpy.Parameter(
            displayName="Dwave Systems API Key",
            name="api_key",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )

        param2 = arcpy.Parameter(
            displayName="TSP Start Point",
            name="tsp_start",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )

        param3 = arcpy.Parameter(
            displayName="Distance Calculation Method",
            name="distance_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        param3.filter.type = "ValueList"
        param3.filter.list = ["Fastest", "Accurate"]

        params = [param0, param1, param2, param3]

        return params

    def updateParameters(self, parameters):
        """Populate the tsp_start dropdown list with OBJECTID values"""
        if parameters[0].value:
            with arcpy.da.SearchCursor(parameters[0].value, ["OBJECTID"]) as cursor:
                parameters[2].filter.list = [row[0] for row in cursor]

    def execute(self, parameters, messages):
        start_time = time.time()

        def check_lists_equal(list1, list2):
            try:
                output = sorted(list1) == sorted(list2) and len(list1) == len(list2)
            except:
                output = False
            arcpy.AddMessage(f"The Route Validity is: {output}")
            return output

        def calculate_distance(x1, y1, x2, y2):
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            return distance

        def create_graph_from_distance_matrix(distance_matrix):
            G = nx.Graph()
            for (id1, id2), distance in distance_matrix.items():
                G.add_edge(id1, id2, weight=distance)
            return G

        def solve_tsp(G, api_token, endpoint_url, start_point):
            arcpy.AddMessage(f"Start TSP Solver at: {start_point}")
            NUM_OF_RUNS = 500
            try:
                start_time_tsp = time.time()
                sampler = EmbeddingComposite(
                    DWaveSampler(
                        token=api_token, endpoint=endpoint_url, num_reads=NUM_OF_RUNS
                    )
                )
                route = dnx.traveling_salesperson(
                    G, sampler, start=start_point, lagrange=400
                )
                end_time_tsp = time.time()
                elapsed_time = end_time_tsp - start_time_tsp

                arcpy.AddMessage(f"Time taken to solve TSP: {elapsed_time} seconds")
            except Exception as e:
                return list()
            return route

        def create_route_segments(
            route, point_data, split_polyline_feature_class, gdb_path
        ):
            feature_class_path = f"{gdb_path}\\{split_polyline_feature_class}"
            arcpy.CreateFeatureclass_management(
                gdb_path, split_polyline_feature_class, "POLYLINE"
            )
            arcpy.AddField_management(feature_class_path, "SegmentID", "SHORT")
            with arcpy.da.InsertCursor(
                feature_class_path, ["SHAPE@", "SegmentID"]
            ) as cursor:
                for i in range(len(route) - 1):
                    start_point = arcpy.Point(
                        *[p[:2] for p in point_data if p[2] == route[i]][0]
                    )
                    end_point = arcpy.Point(
                        *[p[:2] for p in point_data if p[2] == route[i + 1]][0]
                    )
                    point_array = arcpy.Array([start_point, end_point])
                    polyline = arcpy.Polyline(point_array)
                    cursor.insertRow([polyline, i])

                last_point = arcpy.Point(
                    *[p[:2] for p in point_data if p[2] == route[-1]][0]
                )
                first_point = arcpy.Point(
                    *[p[:2] for p in point_data if p[2] == route[0]][0]
                )
                point_array = arcpy.Array([last_point, first_point])
                polyline = arcpy.Polyline(point_array)
                cursor.insertRow([polyline, len(route) - 1])

        arcpy.env.overwriteOutput = True
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        map = aprx.listMaps()[0]
        gdb_path = aprx.defaultGeodatabase

        input_layer = parameters[0].valueAsText
        api_token = parameters[1].valueAsText
        start_point = int(parameters[2].valueAsText)
        distance_method = parameters[3].valueAsText

        split_polyline_feature_class = "tsp_solved_route"

        endpoint_url = "https://eu-central-1.cloud.dwavesys.com/sapi/v2/"

        point_data = []
        distance_matrix = {}

        with arcpy.da.SearchCursor(
            input_layer, ["SHAPE@X", "SHAPE@Y", "OID@"]
        ) as cursor:
            for row in cursor:
                point_data.append(row)
        if distance_method == "Fastest":
            for i, (x1, y1, id1) in enumerate(point_data):
                for j, (x2, y2, id2) in enumerate(point_data):
                    distance_matrix[(id1, id2)] = calculate_distance(x1, y1, x2, y2)

        elif distance_method == "Accurate":
            if arcpy.CheckExtension("network") == "Available":
                arcpy.CheckOutExtension("network")
            else:
                raise arcpy.ExecuteError(
                    "Network Analyst Extension license is not available."
                )
            out_na_layer_name = "ODCostMatrixTSP"
            travel_mode = "Driving Distance"
            search_tolerance = "10000 Meters"
            out_odcm_fc = f"{gdb_path}\\ODCostMatrix"
            origins = input_layer
            destinations = input_layer
            network_dataset = "https://www.arcgis.com/"
            output_layer_file = f"{out_odcm_fc}.lyrx"

            result_object = arcpy.na.MakeODCostMatrixAnalysisLayer(
                network_dataset, out_na_layer_name, travel_mode
            )

            out_na_layer = result_object.getOutput(0)
            sublayer_names = arcpy.na.GetNAClassNames(out_na_layer)
            origins_layer_name = sublayer_names["Origins"]
            destinations_layer_name = sublayer_names["Destinations"]

            arcpy.na.AddLocations(
                out_na_layer, origins_layer_name, origins, "", search_tolerance
            )

            arcpy.na.AddLocations(
                out_na_layer,
                destinations_layer_name,
                destinations,
                "",
                search_tolerance,
            )

            arcpy.na.Solve(out_na_layer)
            out_na_layer.saveACopy(output_layer_file)

            distance_matrix = {}

            route_specialties_table = out_na_layer.listLayers("Lines")[0]
            arcpy.AddMessage("Solved Distance Matrix with OD Cost Matrix")

            with arcpy.da.SearchCursor(
                route_specialties_table,
                ["OriginID", "DestinationID", "Total_Kilometers"],
            ) as cursor:
                for row in cursor:
                    origin = row[0]
                    destination = row[1]
                    distance = row[2]
                    distance_matrix[(origin, destination)] = distance

        G = create_graph_from_distance_matrix(distance_matrix)
        route = solve_tsp(G, api_token, endpoint_url, start_point)
        valid_route = check_lists_equal(G.nodes, route)
        arcpy.AddMessage(f"Calculated Route: {route}")

        counter = 0
        max_runs = 5
        while valid_route is False and counter < max_runs:
            route = solve_tsp(G, api_token, endpoint_url, start_point)
            arcpy.AddMessage(f"Calculated Route: {route}")
            valid_route = check_lists_equal(G.nodes, route)
            counter += 1

            if counter >= max_runs:
                arcpy.AddMessage("Maximum number of runs reached.")
                break

        arcpy.AddMessage(f"Optimum Route: {route}")

        if route is not None:
            if distance_method == "Accurate":
                total_length = 0.0
                segment_layers = []
                start_time_routing = time.time()
                for i in range(len(route) - 1):
                    temp_stops_fc = f"{gdb_path}\\TempStops{i}"
                    arcpy.CreateFeatureclass_management(
                        gdb_path,
                        f"TempStops{i}",
                        "POINT",
                    )
                    arcpy.AddField_management(temp_stops_fc, "StopID", "SHORT")
                    cursor = arcpy.da.InsertCursor(temp_stops_fc, ["SHAPE@", "StopID"])
                    for j, point_id in enumerate(route[i : i + 2]):
                        point = [p for p in point_data if p[2] == point_id][0]
                        cursor.insertRow([arcpy.Point(point[0], point[1]), j + 1])
                    del cursor
                    out_route_layer = arcpy.na.MakeRouteAnalysisLayer(
                        network_dataset,
                        "Route",
                        "Driving Time",
                        line_shape="ALONG_NETWORK",
                    ).getOutput(0)
                    sublayer_names = arcpy.na.GetNAClassNames(out_route_layer)
                    arcpy.na.AddLocations(
                        out_route_layer, sublayer_names["Stops"], temp_stops_fc, ""
                    )
                    arcpy.na.Solve(out_route_layer)

                    route_sublayer = out_route_layer.listLayers("Routes")[0]
                    segment_fc_name = f"Segment_{i}"
                    arcpy.management.CopyFeatures(
                        route_sublayer, f"{gdb_path}\\{segment_fc_name}"
                    )
                    segment_layers.append(f"{gdb_path}\\{segment_fc_name}")

                    with arcpy.da.SearchCursor(
                        f"{gdb_path}\\{segment_fc_name}", ["Total_Kilometers"]
                    ) as cursor:
                        for row in cursor:
                            arcpy.AddMessage(
                                f"Total Length of Segment: {total_length}."
                            )
                            total_length += row[0]

                    map.addDataFromPath(f"{gdb_path}\\{segment_fc_name}")

                end_time_routing = time.time()
                elapsed_time_routing = end_time_routing - start_time_routing
                arcpy.AddMessage(f"Total Length of Route: {total_length}.")

                arcpy.AddMessage(
                    f"Time taken to solve road route with ArcGIS: {elapsed_time_routing} seconds"
                )

            else:
                create_route_segments(
                    route, point_data, split_polyline_feature_class, gdb_path
                )
                arcpy.AddMessage(
                    f"Created route polyline segments in {split_polyline_feature_class}"
                )

                layer = map.addDataFromPath(
                    f"{gdb_path}\\{split_polyline_feature_class}"
                )
                layer.showLabels = True
                label_class = layer.listLabelClasses()[0]
                label_class.expression = "$feature.OBJECTID"
                label_class.expressionType = "arcade"
            end_time = time.time()
            elapsed_time = end_time - start_time

        arcpy.AddMessage(f"Time taken to Complete: {elapsed_time} seconds")


class StructuralImbalance(object):

    def __init__(self):
        """
        Define the tool (the tool name is the name of the class)
        """
        self.label = "Quantum SI Solver"
        self.description = "Find Structural Imbalance using a D-Wave quantum \
computer"
        self.canRunInBackground = False

        self.log_enabled = False
        self.kg = None
        self.G = None
        self.solution = None

    def getParameterInfo(self):
        """Define parameter definitions."""
        param0 = arcpy.Parameter(
            displayName="Input JSON File",
            name="input_json_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )
        param0.filter.list = ["json"]

        param1 = arcpy.Parameter(
            displayName="SAPI Token",
            name="sapi_token",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        param2 = arcpy.Parameter(
            displayName="Graph URL",
            name="graph_url",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        param3 = arcpy.Parameter(
            displayName="Entity types",
            name="entity_types",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True,
        )
        param3.filter.type = "ValueList"
        param3.filter.list = ["Loading..."]

        param4 = arcpy.Parameter(
            displayName="Positive List",
            name="positive_list",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True,
        )
        param4.filter.type = "ValueList"
        param4.filter.list = ["Loading..."]

        # Negative list parameter
        param5 = arcpy.Parameter(
            displayName="Negative List",
            name="negative_list",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True,
        )
        param5.filter.type = "ValueList"
        param5.filter.list = ["Loading..."]

        param6 = arcpy.Parameter(
            displayName="Use hybrid solver",
            name="use_hybrid",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
        )
        param6.value = False

        param7 = arcpy.Parameter(
            displayName="Log Enabled",
            name="log_enabled",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input",
        )
        param7.value = False

        """
        This is an unfortunate hack, but we can't store auth or kg in the class
        itself, because we need to declare them as None in __init__ to pass the
        syntax checker, and this appears to wipe them between updateParameters
        and updateMessages
        """
        param8 = arcpy.Parameter(
            displayName="Authentication error",
            name="auth_error",
            datatype="GPString",
            parameterType="Derived",
            direction="Output",
        )
        param8.value = ""

        return [
            param0, param1, param2, param3, param4,param5, param6, param7,
            param8
        ]
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation."""

        use_input_file = False
        if parameters[0].altered:
            input_file = parameters[0].valueAsText
            if input_file:
                try:
                    with open(input_file, 'r') as file:
                        data = json.load(file)
                        parameters[1].value = data.get('sapi_token', '')
                        parameters[2].value = data.get('graph_url', '')
                        parameters[3].value = data.get('entity_types', [])
                        parameters[4].value = data.get('positive_list', [])
                        parameters[5].value = data.get('negative_list', [])
                        parameters[6].value = data.get('use_hybrid', False)
                        parameters[7].value = data.get('log_enabled', False)
                        use_input_file = True
                except Exception as e:
                    arcpy.AddError(f"Error reading input file: {e}")
        for param in parameters[1:]:
            param.enabled = not use_input_file

        parameters[8].value = ""
        if parameters[2].altered:
            self.auth = GIS("pro")
            if not self.auth:
                parameters[8].value = "Failed to authenticate; \
are you signed in on a domain that has access to Esri Knowledge?"
            else:
                kg = KnowledgeGraph(parameters[2].valueAsText, gis=self.auth)
                if not kg:
                    parameters[8].value = f"Failed to open graph; is there a graph \
    at {parameters[2].valueAsText} and do you have permission to open it?"
                else:
                    all_entity_types = kg.datamodel[
                        "entity_types"].keys()
                    parameters[3].filter.list = list(all_entity_types)
                    all_relationship_types = kg.datamodel[
                        "relationship_types"].keys()
                    parameters[4].filter.list = list(all_relationship_types)
                    parameters[5].filter.list = list(all_relationship_types)

    def updateMessages(self, parameters):
        for p in parameters:
            p.clearMessage()

        def target(index):
            return parameters[0 if parameters[0].altered else index]

        if parameters[8].valueAsText:
            target(2).setErrorMessage(parameters[8].valueAsText)
            return # don't overwrite this error with another- this one is more important

        if parameters[3].altered:
            entity_types = parameters[3].valueAsText
            if not entity_types: 
                target(3).setErrorMessage(
                    "Entity type list must not be empty.")

        if parameters[4].altered or parameters[5].altered:
            p_list = parameters[4].valueAsText
            n_list = parameters[5].valueAsText
            if not p_list:
                target(4).setErrorMessage(
                    "Positive list must not be empty.")
            elif not n_list:
                target(5).setErrorMessage(
                    "Negative list must not be empty.")
            else:
                if not set(p_list.split(";")).isdisjoint(n_list.split(";")):
                    target(4).setErrorMessage(
                        "Positive & negative lists must not share any items")

    def execute(self, parameters, messages):

        self.kg = KnowledgeGraph(parameters[2].valueAsText, gis=GIS("pro"))

        sapi_token = parameters[1].valueAsText
        #graph_url = parameters[2].valueAsText
        entity_types = parameters[3].valueAsText.split(";")
        positive_list = parameters[4].valueAsText.split(";")
        negative_list = parameters[5].valueAsText.split(";")
        self.use_hybrid = parameters[6].value
        self.log_enabled = parameters[7].value

        self.G = nx.Graph()
        self.add_relationships(entity_types, positive_list, 1)
        self.add_relationships(entity_types, negative_list, -1)
        self.solve(sapi_token)
        self.process_solution(entity_types)

        self.log_enabled = False
        self.kg = None
        self.G = None
        self.solution = None

    def add_relationships(self, entity_types, list_items, sign):
        formatted_list_items = ""
        for index, item in enumerate(list_items):
            formatted_list_items += (":" if index == 0 else "|") + item
        formatted_e1 = ""
        formatted_e2 = ""
        for index, item in enumerate(entity_types):
            formatted_e1 += ("e1:" if index == 0 else " OR e1:") + item
            formatted_e2 += ("e2:" if index == 0 else " OR e2:") + item
        query = f'\
MATCH (e1)-[r{formatted_list_items}]->(e2) \
WHERE ({formatted_e1}) AND {formatted_e2} \
RETURN e1,e2'
        if self.log_enabled:
            arcpy.AddMessage("Submitting query:\n" + query)
        query_results = self.kg.query(query)
        formatted_results = [[{
            "type": q[0]["_typeName"],
            "id": str(q[0]["_properties"]["globalid"]),
        },{
            "type": q[1]["_typeName"],
            "id": str(q[1]["_properties"]["globalid"]),
        }] for q in query_results]
        if self.log_enabled:
            arcpy.AddMessage("Query returned:\n" + str(formatted_results))
        for e1, e2 in formatted_results:
            if e1["id"] not in self.G.nodes:
                self.G.add_node(e1["id"], type=e1["type"])
            if e2["id"] not in self.G.nodes:
                self.G.add_node(e2["id"], type=e2["type"])
            self.G.add_edge(e1["id"], e2["id"], sign=sign)

    def solve(self, sapi_token):
        is_valid = False
        attempt = 0
        MAX_ATTEMPTS = 5
        RUN_COUNT = 500
        while not is_valid and attempt < MAX_ATTEMPTS:
            solution = self.solve_once(sapi_token, RUN_COUNT)
            if self.log_enabled:
                arcpy.AddMessage(f"Found solution: {solution}")
            is_valid = self.validate_solution(solution)
            attempt += 1

        if not is_valid:
            raise arcpy.ExecuteError(
                f"No solution found after {MAX_ATTEMPTS} attempts"
            )

        self.solution = solution
    
    def solve_once(self, sapi_token, run_count):
        try:
            start_time_si = time.time()

            if self.use_hybrid:
                sampler = LeapHybridSampler(
                    token=sapi_token,
                    endpoint=self.endpoint()
                )
            else:
                sampler = EmbeddingComposite(DWaveSampler(
                    token=sapi_token,
                    endpoint=self.endpoint(),
                    num_reads=run_count
                ))
            imbalance = dnx.structural_imbalance(self.G, sampler)

            end_time_si = time.time()
            elapsed_time = end_time_si - start_time_si
            if self.log_enabled:
                arcpy.AddMessage(f"SI solved in {elapsed_time} seconds")

            return imbalance
        except Exception as e:
            raise arcpy.ExecuteError("Exception while solving SI: " + str(e))

    def endpoint(self):
        return "https://" \
            + ("na-west-1" if self.use_hybrid else "eu-central-1") \
            + ".cloud.dwavesys.com/sapi/v2/"

    def validate_solution(self, candidate):
        return True  # TODO

    def process_solution(self, entity_types):
        GROUP_AFFILIATION_PROPERTY = "q_group"
        FRUSTRATED_RELATIONS_TYPE = "q_frustrated"

        frustrated_relation = [{
            "name": FRUSTRATED_RELATIONS_TYPE,
            "properties": {
                "status": {
                    "name": "status",
                    "role": "esriGraphPropertyRegular"
                }
            },
        }]
        group_property = [
            {
                "name": GROUP_AFFILIATION_PROPERTY,
                "alias": GROUP_AFFILIATION_PROPERTY,
                "fieldType": "esriFieldTypeSmallInteger",
                "nullable": True,
                "editable": True,
                "defaultValue": 0,
                "visible": True,
                "required": False,
                "isSystemMaintained": False,
                "role": "esriGraphPropertyRegular",
            }
        ]

        # Find groups that have the quantum group property which don't need it anymore
        unused_entities_with_group = [
            e for e in self.kg.datamodel["entity_types"] \
                if GROUP_AFFILIATION_PROPERTY in self.kg.datamodel["entity_types"][e]["properties"].keys() \
                    and e not in entity_types
        ]

        # Find old frustrated relationships that aren't needed anymore
        old_frustrated_relationships = self.kg.query(
            f"MATCH ()-[r:{FRUSTRATED_RELATIONS_TYPE}]->() RETURN r.globalid"
        )
        if len(old_frustrated_relationships) == 0:
            old_frustrated_relationships = []
        else:
            old_frustrated_relationships = [
                {
                    "_objectType": "relationship",
                    "_typeName": FRUSTRATED_RELATIONS_TYPE,
                    "_ids": [u[0] for u in old_frustrated_relationships],
                }
            ]

        # Make a list of the new frustrated relationships to be added
        new_frustrated_relationships = [
            {
                "_objectType": "relationship",
                "_typeName": FRUSTRATED_RELATIONS_TYPE,
                "_originEntityId": UUID(e_pair[0]),
                "_destinationEntityId": UUID(e_pair[1]),
                "_properties": {
                    "status": "positive" if sign["sign"] == 1 else "negative"
                },
            }
            for e_pair, sign in self.solution[0].items()
        ]

        # Make a list of the quantum group properties to be set
        groups = [
            {
                "_objectType": "entity",
                "_typeName": self.G.nodes[entity]["type"],
                "_id": UUID(entity),
                "_properties": {GROUP_AFFILIATION_PROPERTY: group_number},
            }
            for entity, group_number in self.solution[1].items()
        ]

        if self.log_enabled:
            arcpy.AddMessage(
                "Entities with existing property to be removed:\n"
                + str(unused_entities_with_group)
            )
            arcpy.AddMessage(
                "New frustrated relation:\n" + str(frustrated_relation)
            )
            arcpy.AddMessage("New group property:\n" + str(group_property))
            arcpy.AddMessage(
                "New frustrated relationships:\n"
                + str(new_frustrated_relationships)
            )
            arcpy.AddMessage(
                "Old frustrated relationships:\n"
                + str(old_frustrated_relationships)
            )
            arcpy.AddMessage("Affiliation groups:\n" + str(groups))

        for entity_with_old_property in unused_entities_with_group:
            self.kg.graph_property_delete(
                entity_with_old_property, GROUP_AFFILIATION_PROPERTY)
        if (
            not FRUSTRATED_RELATIONS_TYPE
            in self.kg.datamodel["relationship_types"].keys()
        ):
            add_relation_result = self.kg.named_object_type_adds(
                relationship_types=frustrated_relation
            )["relationshipAddResults"]
            if self.log_enabled:
                arcpy.AddMessage(
                    "Add relation result:\n" + str(add_relation_result)
                )
            if len(add_relation_result) > 0:
                self.parse_for_error(add_relation_result[0])
        for entity_type in entity_types:
            if (
                not GROUP_AFFILIATION_PROPERTY
                in self.kg.datamodel["entity_types"][entity_type]["properties"].keys()
            ):
                add_property_result = self.kg.graph_property_adds(
                    entity_type, group_property
                )["propertyAddResults"]
                if self.log_enabled:
                    arcpy.AddMessage(
                        "Add property result:\n" + str(add_property_result)
                    )
                if len(add_property_result) > 0:
                    self.parse_for_error(add_property_result[0])
        apply_edits_result = self.kg.apply_edits(
            adds=new_frustrated_relationships,
            updates=groups,
            deletes=old_frustrated_relationships,
        )
        if self.log_enabled:
            arcpy.AddMessage("Apply edits result:\n" + str(apply_edits_result))
        self.parse_for_error(apply_edits_result)

    def parse_for_error(self, result):
        if "error" in result.keys():
            error = result["error"]
            raise arcpy.ExecuteError(
                "Error "
                + error["error_message"]
                + ", code "
                + str(error["error_code"])
            )
