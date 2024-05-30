# graph-streamlit
This repository contains work on graph construction and UI interaction. It utilizes Neo4j sandbox as the graph database container and Streamlit for the user interface.

## Neo4j connection information
Since the Neo4j sandbox currently permits free use for a maximum of 17 days (7 + 10), the connection information must be updated once it expires. Go to the settings in the Streamlit cloud and update the Neo4j sandbox connection information as follows:

```
NEO4J_SANDBOX = "bolt://52.1.51.217:7687"
NEO4J_SANDBOX_USER = "neo4j"
NEO4J_SANDBOX_PASSWORD = "addressees-huts-pastes"
```