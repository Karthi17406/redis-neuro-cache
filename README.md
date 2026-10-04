#  Redis NeuroCache

### An intelligent Redis-inspired in-memory database with ML-powered cache eviction

**Redis NeuroCache** is a Redis-inspired key-value database built from scratch in **Java**, extended with an **ML-powered eviction system** that predicts which cached keys are less likely to be requested again.

Instead of relying only on traditional policies such as **LRU** or **LFU**, NeuroCache uses a trained **Random Forest model** to estimate the probability that a key will be reused soon.

When the ML service is unavailable, NeuroCache automatically falls back to **LRU**, keeping the cache operational instead of depending entirely on the ML system.

---

##  Why NeuroCache?

Traditional cache eviction policies use fixed rules:

* **LRU** → remove the least recently used key
* **LFU** → remove the least frequently used key
* **NONE** → don't automatically evict

These policies work well, but they don't learn from access behavior.

NeuroCache introduces an additional layer:

```text
                ┌─────────────────────┐
                │     Mini Redis      │
                │      Database       │
                └──────────┬──────────┘
                           │
                     Cache is full
                           │
                           ▼
                ┌─────────────────────┐
                │   Eviction Manager  │
                └──────────┬──────────┘
                           │
                  Policy = ML
                           │
                           ▼
                ┌─────────────────────┐
                │      MLClient       │
                │       (Java)        │
                └──────────┬──────────┘
                           │ HTTP
                           ▼
                ┌─────────────────────┐
                │   Python ML API     │
                │      FastAPI        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Random Forest     │
                │       Model         │
                └──────────┬──────────┘
                           │
                  Reuse probabilities
                           │
                           ▼
                ┌─────────────────────┐
                │ Lowest probability  │
                │      → evict        │
                └─────────────────────┘
```

The result is a cache that can use **observed access behavior** rather than relying exclusively on a fixed eviction rule.

---

#  Features

## Core Redis-like functionality

* `SET`
* `GET`
* `DELETE`
* `EXISTS`
* `KEYS`
* `FLUSH`
* `EXPIRE`
* `TTL`
* `HELP`

## Transactions

* `BEGIN`
* `COMMIT`
* `ROLLBACK`
* Per-client transaction isolation

## Persistence

* Persistent database storage
* Access logging
* Data recovery between runs

## Concurrency

* Multi-client TCP server
* Thread-pool based client handling
* Synchronized database operations
* Tested with simultaneous clients

## Cache eviction

Supports multiple eviction policies:

```text
LRU
LFU
ML
NONE
```

## ML-powered eviction

The ML system:

1. Collects key access statistics
2. Converts them into features
3. Sends features to the ML service
4. Predicts reuse probability
5. Ranks candidate keys
6. Evicts keys with lower predicted reuse probability

## Fault tolerance

If the ML service becomes unavailable:

```text
ML Service
    ↓
request fails
    ↓
Circuit breaker / failure handling
    ↓
Fallback to LRU
    ↓
Cache continues operating
```

The cache does not need the ML service to remain available in order to enforce its configured capacity.

---

#  Architecture

```text
                         Client
                           │
                           ▼
                    ┌─────────────┐
                    │ RedisClient │
                    └──────┬──────┘
                           │ TCP
                           ▼
                    ┌─────────────┐
                    │ RedisServer │
                    └──────┬──────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ ClientHandler│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Command      │
                    │ Executor     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Database   │
                    └──────┬───────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
        ┌────────────────┐   ┌─────────────────┐
        │ Stats Tracker  │   │ EvictionManager │
        └────────────────┘   └────────┬────────┘
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                         ▼            ▼            ▼
                        LRU          LFU           ML
                                                   │
                                                   ▼
                                              ┌─────────┐
                                              │ MLClient │
                                              └────┬────┘
                                                   │ HTTP
                                                   ▼
                                              ┌─────────┐
                                              │ FastAPI │
                                              └────┬────┘
                                                   │
                                                   ▼
                                           Random Forest
                                                   │
                                                   ▼
                                          Reuse Probability
```

---

# Machine Learning Layer

The central idea behind the ML eviction system is:

> **Predict whether a key is likely to be requested again soon.**

For each key, NeuroCache extracts behavioral information such as:

| Feature                      | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `total_accesses`             | Total number of accesses                       |
| `get_count`                  | Number of GET operations                       |
| `set_count`                  | Number of SET operations                       |
| `age_seconds`                | Age of the key                                 |
| `recency_seconds`            | Time since its most recent access              |
| `mean_interval_seconds`      | Average time between accesses                  |
| `ewma_interval_seconds`      | Exponentially weighted access interval         |
| `accesses_per_minute`        | Access frequency                               |
| `get_ratio`                  | GET operations relative to total operations    |
| `value_size`                 | Size of the stored value                       |
| `recency_over_mean_interval` | Recency relative to historical access interval |

The model produces a value between `0` and `1`:

```text
0.05  → very low predicted reuse
0.30  → relatively low predicted reuse
0.65  → higher predicted reuse
0.90  → very high predicted reuse
```

During eviction, keys with lower predicted reuse probability become stronger eviction candidates.

---

#  Example

Suppose the cache contains:

```text
hot_key
warm_key
cold_key
```

The ML model might produce:

```text
hot_key   → 0.63
warm_key  → 0.45
cold_key  → 0.09
```

NeuroCache ranks them:

```text
cold_key
   ↓
warm_key
   ↓
hot_key
```

If one key needs to be evicted:

```text
cold_key → EVICT
```

The important distinction is that the decision is based on the model's predicted reuse probability rather than simply looking at one statistic such as "last access time" or "access count."

---

# ML Failure Handling

Machine learning should not become a single point of failure.

NeuroCache therefore follows this design:

```text
              Cache reaches capacity
                       │
                       ▼
                 Policy = ML
                       │
                       ▼
                 ML Service?
                  /        \
                YES         NO
                │            │
                ▼            ▼
           ML prediction   LRU fallback
                │            │
                └──────┬─────┘
                       ▼
                  Evict key
```

If the ML service is unavailable:

```text
ML request
    ↓
Failure
    ↓
Fallback decision
    ↓
LRU
```

A circuit-breaker mechanism prevents every subsequent cache write from repeatedly waiting for an unavailable ML service.

---

# 📊 Model Information

The current trained model is a **Random Forest classifier**.

The service exposes model information through:

```text
GET /model_info
```

The model was trained to predict whether a key would be reused within a short future horizon.

The repository also contains the generated model artifacts under:

```text
src/models/
```

Example:

```text
src/models/
├── eviction_model.joblib
└── eviction_model_evict.joblib
```

---

#  ML Service API

The ML layer is implemented as a Python service using **FastAPI**.

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "random_forest",
  "feature_count": 11
}
```

### Model information

```http
GET /model_info
```

### Prediction

```http
POST /predict
```

### Eviction ranking

```http
POST /evict
```

The `/evict` endpoint receives candidate keys and returns their predicted reuse probabilities along with the keys selected as eviction victims.

---

# Project Structure

```text
Redis-NeuroCache/
│
├── src/
│   │
│   ├── Main.java
│   ├── RedisServer.java
│   ├── RedisClient.java
│   ├── ClientHandler.java
│   ├── CommandExecutor.java
│   ├── Database.java
│   │
│   ├── EvictionManager.java
│   ├── EvictionPolicy.java
│   ├── MLClient.java
│   ├── KeyStats.java
│   ├── StatsTracker.java
│   │
│   ├── PersistenceManager.java
│   ├── TransactionManager.java
│   │
│   ├── TestEviction.java
│   ├── TestMLClient.java
│   └── ...
│
├── src/
│   ├── ml/
│   │   ├── service.py
│   │   ├── train.py
│   │   ├── features.py
│   │   ├── data_loader.py
│   │   ├── config.py
│   │   └── build_dataset.py
│   │
│   ├── models/
│   │   ├── eviction_model.joblib
│   │   └── eviction_model_evict.joblib
│   │
│   └── data/
│       ├── dataset_evict.csv
│       ├── ml_dataset.csv
│       ├── workload_log.csv
│       └── ...
│
├── backup_before_ml/
├── .vscode/
└── README.md
```

---

#  Requirements

### Java

* JDK 17+ recommended

### Python

* Python 3.10+
* FastAPI
* Uvicorn
* NumPy
* Pydantic
* Joblib
* XGBoost / ML dependencies used by the training pipeline

---

#  Running the ML Service

From the project root:

```powershell
cd D:\MiniRedisNew\MiniRedisNew
```

Set the Python source path:

```powershell
$env:PYTHONPATH="$PWD\src"
```

Start the ML service:

```powershell
python -m ml.service
```

The service starts at:

```text
http://127.0.0.1:8000
```

Verify it:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected:

```text
status        : ok
model_loaded  : True
model_name    : random_forest
feature_count : 11
```

---

# Running the Java Application

Compile the Java source files according to the project's build setup and run the server.

The Redis server uses:

```text
Port: 6380
```

A client can then connect to the server and execute commands such as:

```text
SET name Karthika
GET name
EXISTS name
KEYS
DELETE name
```

---

#  Enabling ML Eviction

The Java ML client can be configured with JVM properties:

```text
-Dminiredis.ml.enabled=true
-Dminiredis.ml.url=http://127.0.0.1:8000
-Dminiredis.ml.timeoutms=250
```

Example:

```text
ML enabled
     ↓
http://127.0.0.1:8000
     ↓
Random Forest
```

When ML is disabled, the system can operate without the ML service.

---

# Testing

NeuroCache includes dedicated tests for the eviction subsystem.

Run:

```text
TestEviction
```

The test suite covers:

### 1. Unbounded database

Verifies that a database without a maximum key limit behaves normally.

### 2. Capacity enforcement

Verifies that the configured maximum number of keys is respected.

### 3. LRU

Verifies that the least recently used key is selected.

### 4. LFU

Verifies that the least frequently used key is selected.

### 5. NONE

Verifies that the database does not evict keys when eviction is explicitly disabled.

### 6. ML service unavailable

Verifies:

```text
ML failure
    ↓
LRU fallback
    ↓
capacity still enforced
```

### 7. ML service available

Verifies:

```text
Java
 ↓
MLClient
 ↓
FastAPI
 ↓
Random Forest
 ↓
eviction decision
```

and confirms that the ML decision is actually recorded by the eviction manager.

### 8. Statistics cleanup

Verifies that statistics for evicted keys are removed.

### 9. Dynamic capacity changes

Verifies that reducing the maximum capacity immediately evicts excess keys.

---

#  Example ML Prediction

A candidate key can be represented using its behavioral features:

```json
{
  "key": "hot_key",
  "features": {
    "total_accesses": 100,
    "get_count": 90,
    "set_count": 10,
    "age_seconds": 100,
    "recency_seconds": 0.5,
    "mean_interval_seconds": 1.0,
    "ewma_interval_seconds": 0.8,
    "accesses_per_minute": 60,
    "get_ratio": 0.9,
    "value_size": 100,
    "recency_over_mean_interval": 0.5
  }
}
```

The model returns a reuse probability:

```text
hot_key → 0.63
```

A lower score indicates a lower predicted likelihood of near-term reuse.

---

#  Design Principles

## 1. ML should enhance the cache, not break it

The database should continue operating even if the ML service fails.

## 2. Eviction must always respect capacity

For bounded eviction policies:

```text
database size <= configured maximum
```

## 3. Statistics belong to resident keys

When a key is evicted, its associated access statistics are removed as well.

## 4. Traditional policies remain available

ML is an additional strategy, not a replacement for every existing eviction policy.

## 5. Java owns the database

The Java application remains responsible for:

* data storage
* commands
* transactions
* clients
* eviction decisions
* persistence

Python is used specifically for the ML inference layer.

---

# Technology Stack

| Layer                | Technology                      |
| -------------------- | ------------------------------- |
| Core database        | Java                            |
| Networking           | Java TCP sockets                |
| Concurrency          | Java threads / thread pool      |
| Persistence          | File-based storage              |
| Transactions         | Custom transaction manager      |
| ML service           | Python                          |
| API                  | FastAPI                         |
| ML model             | Random Forest                   |
| Model serialization  | Joblib                          |
| Numerical processing | NumPy                           |
| Communication        | HTTP / JSON                     |
| Testing              | Java test classes + API testing |

---

#  What Makes This Project Different?

This project combines several systems concepts into one application:

```text
        ┌────────────────────────────┐
        │      Distributed-System    │
        │          Concepts           │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │       TCP Networking       │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │       Concurrent Server    │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      Key-Value Store       │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      Cache Eviction        │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │       ML Prediction        │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      Fault Tolerance       │
        └────────────────────────────┘
```

It brings together:

* Data structures
* Networking
* Operating-system concepts
* Multithreading
* Databases
* Caching
* Machine learning
* REST APIs
* Fault tolerance
* Software testing

---

#  Future Improvements

Potential extensions include:

* [ ] Online model retraining
* [ ] More advanced workload generation
* [ ] Redis protocol compatibility
* [ ] Distributed cache nodes
* [ ] Replication
* [ ] Sharding
* [ ] Prometheus metrics
* [ ] Grafana dashboard
* [ ] More ML models
* [ ] Reinforcement-learning-based eviction
* [ ] Benchmarking against Redis
* [ ] Dockerized Java + ML deployment
* [ ] Kubernetes deployment
* [ ] Automated model evaluation
* [ ] Real-time eviction analytics dashboard

---

# Project Goal

The long-term goal of Redis NeuroCache is to explore whether **machine learning can make cache eviction decisions more adaptive to workload behavior while preserving the reliability and predictability expected from a traditional cache.**

Rather than replacing proven cache strategies, NeuroCache treats ML as an intelligent decision layer with a traditional fallback.

---

#  Author

**Karthikayini Srinivasan**

Computer Science Engineering Student

Interested in:

* Backend Engineering
* Distributed Systems
* Databases
* Machine Learning
* System Design
* Scalable Software

---

# ⭐ If you find this project interesting

Feel free to explore the architecture, experiment with different eviction policies, modify the ML features, and benchmark the system against traditional LRU/LFU strategies.

**Redis NeuroCache — making cache eviction behavior-aware.**
