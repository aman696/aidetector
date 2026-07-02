# Monitoring & Alerting

A self-hosted Prometheus + Grafana + Alertmanager stack that watches the whole
server (host resources and every Docker container on it) and emails you when
something needs attention. It runs **alongside** Coolify, not through it —
Coolify manages the app containers; this stack observes them from outside, the
same way it would observe any container on the box.

Everything here is **private-only** by design (see Security below). None of it
should ever get a Cloudflare Tunnel public hostname.

## What it answers

| Question | Where |
|---|---|
| How much disk/CPU/RAM is the whole server using? | node-exporter → Grafana "Node Exporter Full" (import ID 1860) |
| How much is each container using, and is one near its limit? | cAdvisor → Grafana "cAdvisor / Docker Container metrics" (import ID 14282) |
| How many people are using the detector right now? | The app's own `detect_inflight` metric → the provisioned "App Metrics" dashboard |
| Am I about to run out of disk from repeated builds? | `DiskSpaceHigh` / `DiskSpaceCritical` alerts → email |
| Is the detector turning users away (overloaded)? | `TooManyUsersBeingTurnedAway` / `DetectorAtCapacitySustained` alerts → email |
| Is a container about to get OOM-killed? | `ContainerNearMemoryLimit` alert → email |
| Did a container / the whole server go down? | `ContainerDown` alert → email |

## Setup (on the server, over Tailscale SSH)

```bash
cd monitoring
cp .env.example .env                                    # set GRAFANA_ADMIN_PASSWORD
cp alertmanager/alertmanager.yml.example alertmanager/alertmanager.yml
nano alertmanager/alertmanager.yml                       # put in real SMTP creds + your email
docker compose up -d
```

Then, over the SSH tunnel you already use for Coolify (`ssh -L 9000:localhost:8000 ...`),
add forwards for Grafana and (optionally) Prometheus:

```bash
ssh -L 9000:localhost:8000 -L 3000:localhost:3000 -L 9090:localhost:9090 aman@zero
```

- Grafana: `http://localhost:3000` (login `admin` / the password you set)
- Prometheus: `http://localhost:9090` (raw metrics/alert status, mostly for debugging)

On first Grafana login: **Dashboards → Import** → ID `1860` (Node Exporter Full)
and ID `14282` (cAdvisor) → pick the **Prometheus** datasource (already
provisioned). The **"Human or AI? — App Metrics"** dashboard is provisioned
automatically — no import needed.

## Limiting resource usage ("what power it's allowed to use")

That's configured **per-app in Coolify** (Resource Limits section), not in this
stack — this stack only *observes*, it doesn't cap. You already set the prod
and staging apps to Memory `4G` / CPU `2`. To verify or change them: Coolify →
the app → Resource Limits. Watch the "cAdvisor" dashboard's per-container
memory panel against that limit; `ContainerNearMemoryLimit` fires at 90% of
whatever you've set there.

## Wiring up the app's own metrics (optional, but the "too many users" alerts need it)

Host + container metrics (disk, CPU, RAM) work immediately with zero extra
config. The **app-level** metrics (`detect_inflight`, request outcomes, scan
latency — the actual "how many users right now" signal) need Prometheus to
reach the app container directly, which requires one piece of server-specific
info: find the address.

```bash
# find the running app container's name and its network
docker ps --format '{{.Names}}' | grep -i aidetector
docker inspect <container-name> --format '{{json .NetworkSettings.Networks}}' | python3 -m json.tool
```

Then in `prometheus/prometheus.yml`, uncomment the `humanorai-prod` /
`humanorai-staging` scrape jobs and set the target to
`<container-name>:7860` — but only if Prometheus can reach that network. Since
Prometheus runs with `network_mode: host` here, the simplest fix is to also
attach the `prometheus` service (in `docker-compose.yml`) to the same Docker
network the Coolify app uses, e.g.:

```yaml
  prometheus:
    networks:
      - default
      - coolify        # or whatever `docker network ls` shows for your app
networks:
  coolify:
    external: true
```

Then `docker compose up -d` to apply, and uncomment the two scrape jobs. Once
that target is live, the `app` alert group in `alert_rules.yml` starts firing
for real; until then those two rules simply evaluate to no data (harmless).

## Security

- **Nothing here gets a public tunnel hostname.** Grafana, Prometheus, and
  Alertmanager all bind `127.0.0.1` only (explicitly, even under
  `network_mode: host` — see the `GF_SERVER_HTTP_ADDR` / `--web.listen-address`
  settings in `docker-compose.yml`). Reach them only via an SSH tunnel over
  Tailscale, same boundary as the Coolify dashboard.
- `alertmanager/alertmanager.yml` and `.env` hold real credentials (SMTP
  password, Grafana admin password) — both are gitignored
  (`monitoring/.gitignore`); only the `.example` templates are committed.
- `/metrics` on the app itself is reachable wherever the app is reachable
  (including publicly, on prod). It has no data more sensitive than request
  counts and latency, but as defense in depth it supports an optional
  `METRICS_TOKEN` env var — if set, the route requires
  `Authorization: Bearer <token>` or `?token=` and 404s (not 401) otherwise, so
  it doesn't even confirm the route exists. Set this if the app is ever
  reachable from an untrusted network.

## Files

```
monitoring/
├── docker-compose.yml                        # the stack itself
├── .env.example                              # -> .env (Grafana admin password)
├── prometheus/
│   ├── prometheus.yml                        # scrape targets
│   └── alert_rules.yml                       # alert thresholds
├── alertmanager/
│   └── alertmanager.yml.example               # -> alertmanager.yml (SMTP + recipient)
└── grafana/provisioning/
    ├── datasources/prometheus.yml            # auto-provisioned Prometheus datasource
    └── dashboards/
        ├── dashboard.yml                      # tells Grafana to auto-load JSON here
        └── app-specific.json                  # the one dashboard this repo ships;
                                                #   import 1860 + 14282 for the rest
```

## Tuning the alert thresholds

The numbers in `alert_rules.yml` (disk 80%/90%, memory 90%, CPU 90% for 10m,
etc.) are reasonable starting points for a single laptop-server, not measured
constants — adjust them to your actual hardware and comfort level, the same
way you'd tune any operational threshold. Each rule documents its own
reasoning inline.
