# Distributed Mesh & Service Discovery

Iris is built for location transparency, allowing actors to communicate across node boundaries without special developer logic.

## 🌐 Global Service Discovery

Actors are first-class network services. You can register human-readable names and resolve them across the mesh.

### 1. Register a Local Actor
```python
pid = rt.spawn(my_handler)
rt.register("auth_worker", pid)
```

### 2. Look up a Local Name
```python
target = rt.whereis("auth_worker")
```

### 3. Resolve a Remote Actor
Remote resolution returns a local **proxy** actor. When you send a message to this proxy, the runtime automatically forwards it over TCP to the target host.

#### Python
```python
async def find_remote():
    addr = "192.168.1.5:9000"
    # Resolve returns a proxy PID
    proxy_pid = await rt.resolve_remote_py(addr, "auth_worker")
    if proxy_pid:
        # Send just like a local PID
        rt.send(proxy_pid, b"login")
```

#### Node.js
```javascript
async function findAndQuery() {
    const addr = "192.168.1.5:9000";
    const proxyPid = await rt.resolveRemote(addr, "auth_worker");
    if (proxyPid) {
        rt.send(proxyPid, Buffer.from("login"));
    }
}
```

## 🛡️ Distributed Supervision & Self-Healing

- **Heartbeat monitoring:** Automatic `PING`/`PONG` (0x02/0x03) detects silent failures such as GIL stalls or network partitions.
- **Fail-fast:** The TCP protocol enforces 1MiB payload caps and operation timeouts.
- **Structured System Messages:** `EXIT` messages carry reason codes (`Normal`, `Panic`, `Killed`, `Crash`) and metadata across the network to notify supervisors.

## The Mesh Protocol

Iris nodes use a length-prefixed binary TCP protocol.

| Packet Type | Function | Payload Structure |
| :--- | :--- | :--- |
| `0x00` | **User Message** | `[PID: u64][LEN: u32][DATA: Bytes]` |
| `0x01` | **Resolve Request** | `[LEN: u32][NAME: String]` → returns `[PID: u64]` |
| `0x02` | **Heartbeat (Ping)** | `[Empty]` — Probe remote health |
| `0x03` | **Heartbeat (Pong)** | `[Empty]` — Acknowledge health |
