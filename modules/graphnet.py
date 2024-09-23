from neo4j import GraphDatabase
from modules.logbook import setup_class_logger
from neo4j.exceptions import TransientError
import time


class Graphnet:
    """
    A class to handle interactions with a Neo4j graph database.
    Provides functionality to execute queries, add or update nodes and relationships,
    manage retries for transient errors, and perform database maintenance tasks.
    """

    def __init__(self, uri, username, password):
        """
        Initializes the Graphnet instance.

        :param uri: The connection string for the Neo4j database.
        :param username: The username for authentication.
        :param password: The password for authentication.
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logbook = setup_class_logger("Graphnet", "graphnet.log")

    def execute_query(
        self, query, data=None, parameters={}, max_retries=10, retry_delay=3
    ):
        """
        Executes a given Cypher query with optional parameters, implementing retries for transient errors.

        :param query: The Cypher query to execute.
        :param data: Optional dictionary of parameters for the query.
        :param max_retries: Maximum number of retries for transient errors.
        :param retry_delay: Delay between retries in seconds.
        :return: A list of results obtained from the query.
        :raises RuntimeError: If the maximum number of retries is reached.
        """
        attempt = 0
        while attempt < max_retries:
            with self.driver.session() as session:
                try:
                    if data is not None:
                        # Assuming data is a list of dictionaries
                        parameters["data"] = data
                    result = session.run(query, parameters)

                    return [record for record in result]

                except TransientError as e:
                    if "DeadlockDetected" in e.code():
                        time.sleep(retry_delay)
                        attempt += 1
                    else:
                        raise  # Reraise if it's not a deadlock error

        raise RuntimeError(f"Max retry attempts reached for query: {query}")

    def lookup_node(self, object_type, param: dict):
        """
        Looks up a node of a given type with specified parameters.

        :param object_type: The type (label) of the node to search for.
        :param param: A dictionary of parameters to match against the node's properties.
        :return: The first node found matching the criteria or None if no match is found.
        """
        # Use parameterized Cypher query to prevent injection and improve readability
        check_query = f"MATCH (n:{object_type}) WHERE "
        check_query += " AND ".join([f"n.{key} = ${key}" for key in param.keys()])
        check_query += " RETURN n"

        existing = self.execute_query(check_query, param)
        if existing:
            return existing[0]["n"]
        else:
            return None

    def load_data(self, tx):
        pass

    def add_object(self, object_type, object_id, **attributes):
        """
        Adds or updates an object (node) in the database with given attributes.

        :param object_type: The type (label) of the object.
        :param object_id: The ID of the object to add or update.
        :param attributes: Key-value pairs representing the object's properties.
        """
        while True:
            try:
                # Prepare the attribute string for use in Cypher queries
                attribute_string = ", ".join([f"{key}: ${key}" for key in attributes])

                # Check if the object already exists
                check_query = (
                    f"MATCH (n:{object_type}) WHERE n.id = $object_id RETURN n"
                )
                existing = self.execute_query(check_query, {"object_id": object_id})

                if existing:  # Update the existing node
                    update_query = f"MATCH (n:{object_type} {{id: $object_id}}) "
                    update_query += " ".join(
                        [f"SET n.{key} = ${key}" for key in attributes.keys()]
                    )
                    self.execute_query(
                        update_query, {"object_id": object_id, **attributes}
                    )
                else:  # Create a new node
                    create_query = f"CREATE (n:{object_type} {{id: $object_id, {attribute_string}}})"
                    self.execute_query(
                        create_query, {"object_id": object_id, **attributes}
                    )

                break  # Break the loop if operation is successful

            except Exception as e:
                # Log the exception
                self.logbook.exception(f"Error in add_object: {e}")

                # Ask the user what to do next
                response = input(
                    f"Error in add_object: {e}. Press 'r' to retry, 's' to skip, or 'q' to quit: "
                )
                if response.lower() == "r":
                    continue  # Retry
                elif response.lower() == "s":
                    break  # Skip this operation
                elif response.lower() == "q":
                    raise  # Re-raise the exception to quit

    def add_relationship(self, object1, object2, relationship_type, **attributes):
        """
        Adds or updates a relationship between two objects with given attributes.

        :param object1: The ID of the first object.
        :param object2: The ID of the second object.
        :param relationship_type: The type of the relationship to create or update.
        :param attributes: Key-value pairs representing the relationship's properties.
        """
        while True:
            try:
                # Prepare the attribute string for use in Cypher queries
                attribute_string = ", ".join([f"{key}: ${key}" for key in attributes])

                # Check if the relationship already exists
                check_query = f"MATCH (a)-[r:{relationship_type}]-(b) WHERE a.id = $object1 AND b.id = $object2 RETURN r"
                existing = self.execute_query(
                    check_query, {"object1": object1, "object2": object2}
                )

                if existing:  # Update the existing relationship
                    update_query = f"MATCH (a)-[r:{relationship_type}]-(b) WHERE a.id = $object1 AND b.id = $object2 "
                    update_query += " ".join(
                        [f"SET r.{key} = ${key}" for key in attributes.keys()]
                    )
                    self.execute_query(
                        update_query,
                        {"object1": object1, "object2": object2, **attributes},
                    )
                else:  # Create a new relationship
                    create_query = f"MATCH (a), (b) WHERE a.id = $object1 AND b.id = $object2 CREATE (a)-[r:{relationship_type} {{ {attribute_string} }}]->(b)"
                    self.execute_query(
                        create_query,
                        {"object1": object1, "object2": object2, **attributes},
                    )

                break  # Break the loop if operation is successful

            except Exception as e:
                # Log the exception
                self.logbook.exception(f"Error in add_relationship: {e}")

                # Ask the user what to do next
                response = input(
                    f"Error in add_relationship: {e}. Press 'r' to retry, 's' to skip, or 'q' to quit: "
                )
                if response.lower() == "r":
                    continue

    def find_all_duplicate_relationships(self):
        """
        Finds all duplicate relationships in the database.

        :return: A list of dictionaries with relationship types and duplicate relationship IDs.
        """
        duplicate_relationships_query = """
            MATCH (a)-[r]->(b)
            WITH a, b, TYPE(r) as relType, COLLECT(r) as relationships
            WHERE size(relationships) > 1
            WITH relType, relationships[0] as firstRel, relationships[1..] as duplicates
            UNWIND duplicates as dup
            RETURN relType as relationshipType, COLLECT(ID(dup)) as duplicateIds
        """

        with self.driver.session() as session:
            result = session.run(duplicate_relationships_query)
            # Get the list of duplicate relationship IDs and types from the result
            duplicates = [
                {"type": record["relationshipType"], "ids": record["duplicateIds"]}
                for record in result
            ]
            return duplicates

    def delete_relationships_by_id(self, relationship_ids):
        """
        Deletes relationships by their IDs.

        :param relationship_ids: A list of IDs of the relationships to delete.
        """
        delete_query = """
            UNWIND $relationship_ids AS rel_id
            MATCH ()-[r]-()
            WHERE ID(r) = rel_id
            DELETE r"""

        with self.driver.session() as session:
            session.run(delete_query, relationship_ids=relationship_ids)

    def close(self):
        """
        Closes the database connection.
        """
        self.driver.close()

    def purge_database(self):
        """
        Purges the entire database, removing all nodes, relationships, indexes, and constraints.

        WARNING: This action is irreversible and should be used with caution.
        """
        # Retrieve the database name for verification
        db_info_query = (
            "CALL dbms.listConfig('dbms.default_database') YIELD value RETURN value"
        )
        with self.driver.session() as session:
            db_name_record = session.run(db_info_query).single()
            db_name = db_name_record["value"] if db_name_record else "unknown"

        # Ask for user verification
        print(f"WARNING: You are about to purge the database '{db_name}'.")
        user_confirmation = input(f"Please type '{db_name}' to confirm: ")

        if user_confirmation == db_name:
            with self.driver.session() as session:
                # This will delete all nodes and relationships
                delete_all_nodes_and_relationships = "MATCH (n) DETACH DELETE n"
                session.run(delete_all_nodes_and_relationships)

                # Manually drop all indexes and constraints for Neo4j 4.0 and later
                try:
                    session.run("SHOW INDEXES").consume()
                    session.run("SHOW CONSTRAINTS").consume()

                    session.run("CALL apoc.schema.assert({}, {})")
                except Exception as e:
                    print(f"Error dropping indexes and constraints: {e}")

                print(
                    "Database purged. All nodes, relationships, indexes, and constraints have been removed."
                )
        else:
            print("Database purge cancelled.")

    def find_shortest_path(
        self, start_node, end_node, min_strength=50, exclusion_list=[]
    ):
        """
        Finds the shortest path between two nodes in the graph.

        :param start_node: The ID of the start node.
        :param end_node: The ID of the end node.
        :param min_strength: The minimum strength of relationships to consider.
        :return: A list of nodes and relationships representing the shortest path.
        """
        if exclusion_list:
            and_clause = "AND NONE(node IN nodes(path) WHERE node IN $exclusion_list)"

        query = """
        MATCH (start:protein {protein_ID: $start_node}), (end:protein {protein_ID: $end_node})
        MATCH path = shortestPath((start)-[r:combined_score*..12]-(end))
        WHERE ALL(rel IN r WHERE rel.strength >= $min_strength)
        """
        if exclusion_list:
            query += and_clause
        query += "RETURN path"
        # print(query)
        parameters = {
            "start_node": start_node,
            "end_node": end_node,
            "min_strength": min_strength,
            "exclusion_list": exclusion_list,
        }
        return self.execute_query(query, parameters=parameters)

    def check_for_immediate_interactions(self, start_node, end_node, min_strength=50):
        """
        Checks for immediate interactions between two nodes in the graph.

        :param start_node: The ID of the start node.
        :param end_node: The ID of the end node.
        :param min_strength: The minimum strength of relationships to consider.
        :return: A list of nodes and relationships representing the immediate interactions.
        """
        query = """
        MATCH (p1:protein {protein_ID: $start_node}), (p2:protein {protein_ID: $end_node})
        RETURN EXISTS( (p1)-[]-(p2) ) as hasRelationship

        """
        parameters = {
            "start_node": start_node,
            "end_node": end_node,
            "min_strength": min_strength,
        }
        truth = self.execute_query(query, parameters=parameters)
        return truth[0]["hasRelationship"]
