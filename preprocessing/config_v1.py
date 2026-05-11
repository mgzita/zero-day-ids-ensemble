"""
preprocessing/config.py
=======================
Central configuration: dataset paths, constants, and column definitions.
Edit only this file when moving data to a different location.
"""

import os

# =============================================================================
# DATASET PATHS
# =============================================================================

BASE_DIR     = r"C:/MLProject/zero_day_project"
DATA_DIR     = os.path.join(BASE_DIR, "data", "raw")
ARTEFACT_DIR = os.path.join(BASE_DIR, "artefacts")

DATASET_PATHS = {
    "cic2017": os.path.join(DATA_DIR, "CIC2017"),
    "cic2018": os.path.join(DATA_DIR, "CIC2018"),
    "unsw":    os.path.join(DATA_DIR, "UNSW"),
}

# UNSW folder contains metadata files (GT, features) that must be excluded.
# Only these 4 files contain actual flow records.
UNSW_DATA_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

# =============================================================================
# SPLIT SETTINGS  (paper section 4.2 step 4)
# =============================================================================

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# =============================================================================
# FEATURE SELECTION SETTINGS  (paper section 4.2 step 2)
# =============================================================================

CORRELATION_THRESHOLD = 0.85

# =============================================================================
# STRATIFIED SAMPLING  (for large eval datasets like CIC2018 ~16M rows)
# Preserves benign/attack ratio while keeping memory usage manageable.
# Only applied to datasets listed in SAMPLED_DATASETS.
# Increase MAX_EVAL_ROWS if you have more RAM available.
# =============================================================================

MAX_EVAL_ROWS    = 500_000     # max rows to sample from large eval datasets
SAMPLED_DATASETS = ["cic2018"] # datasets to apply stratified sampling to

# =============================================================================
# UNSW-NB15 COLUMN NAMES
# The 4 raw UNSW CSV files contain no header row.
# =============================================================================

UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto",
    "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "service", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "stime", "ltime", "sinpkt", "dinpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]

# =============================================================================
# NON-NUMERIC IDENTIFIER COLUMNS TO DROP DURING CLEANING
#
# Port numbers (sport, dsport, dst_port, destination_port) are intentionally
# kept -- they are numeric and carry useful signal for attack detection.
# =============================================================================

NON_NUMERIC_COLS = [
    "srcip",            # source IP address       -- string
    "dstip",            # destination IP address  -- string
    "proto",            # protocol name           -- string (tcp, udp)
    "state",            # connection state        -- string (CON, INT)
    "service",          # service name            -- string (dns, http)
    "attack_cat",       # attack category         -- string label
    "flow_id",          # flow identifier         -- string
    "source_ip",        # CIC2017 variant
    "destination_ip",   # CIC2017 variant
    "timestamp",        # timestamp               -- not a flow feature
]

# =============================================================================
# LOG TRANSFORM COLUMNS  (supervisor fix 5)
#
# Heavy-tailed network flow features whose distributions differ dramatically
# across datasets. log1p(x) = log(1+x) compresses the right tail and
# stabilises cross-dataset distributions, improving generalisation.
# Only columns present in a given dataset are transformed.
# =============================================================================

LOG_TRANSFORM_COLS = [
    "flow_bytes_s",
    "fwd_bytes_s",
    "bwd_bytes_s",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "total_fwd_packets",
    "total_backward_packets",
    "sbytes",
    "dbytes",
    "fwd_packet_length_max",
    "fwd_packet_length_mean",
    "bwd_packet_length_max",
    "bwd_packet_length_mean",
    "average_packet_size",
    "smeansz",
    "dmeansz",
    "flow_duration",
    "flow_iat_mean",
    "flow_iat_max",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_max",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_max",
    "dur",
    "sinpkt",
    "dinpkt",
    "flow_packets_s",
    "sload",
    "dload",
]

# =============================================================================
# CIC2018 COLUMN RENAMING MAP
#
# CIC2018 uses shorter/different column names for the same features as CIC2017.
# After normalise_columns() is applied, we rename CIC2018 columns to match
# CIC2017 so that get_common_features() finds a meaningful intersection.
#
# Format: "cic2018_normalised_name" -> "cic2017_normalised_name"
# =============================================================================

CIC2018_RENAME = {
    # Traffic volume
    "dst_port":                    "destination_port",
    "tot_fwd_pkts":                "total_fwd_packets",
    "tot_bwd_pkts":                "total_backward_packets",
    "totlen_fwd_pkts":             "total_length_of_fwd_packets",
    "totlen_bwd_pkts":             "total_length_of_bwd_packets",

    # Packet length -- forward
    "fwd_pkt_len_max":             "fwd_packet_length_max",
    "fwd_pkt_len_min":             "fwd_packet_length_min",
    "fwd_pkt_len_mean":            "fwd_packet_length_mean",
    "fwd_pkt_len_std":             "fwd_packet_length_std",

    # Packet length -- backward
    "bwd_pkt_len_max":             "bwd_packet_length_max",
    "bwd_pkt_len_min":             "bwd_packet_length_min",
    "bwd_pkt_len_mean":            "bwd_packet_length_mean",
    "bwd_pkt_len_std":             "bwd_packet_length_std",

    # Flow rate
    "flow_byts_s":                 "flow_bytes_s",
    "flow_pkts_s":                 "flow_packets_s",

    # Inter-arrival times -- flow
    "flow_iat_mean":               "flow_iat_mean",
    "flow_iat_std":                "flow_iat_std",
    "flow_iat_max":                "flow_iat_max",
    "flow_iat_min":                "flow_iat_min",

    # Inter-arrival times -- forward
    "fwd_iat_tot":                 "fwd_iat_total",
    "fwd_iat_mean":                "fwd_iat_mean",
    "fwd_iat_std":                 "fwd_iat_std",
    "fwd_iat_max":                 "fwd_iat_max",
    "fwd_iat_min":                 "fwd_iat_min",

    # Inter-arrival times -- backward
    "bwd_iat_tot":                 "bwd_iat_total",
    "bwd_iat_mean":                "bwd_iat_mean",
    "bwd_iat_std":                 "bwd_iat_std",
    "bwd_iat_max":                 "bwd_iat_max",
    "bwd_iat_min":                 "bwd_iat_min",

    # Flags
    "fwd_psh_flags":               "fwd_psh_flags",
    "bwd_psh_flags":               "bwd_psh_flags",
    "fwd_urg_flags":               "fwd_urg_flags",
    "bwd_urg_flags":               "bwd_urg_flags",
    "fin_flag_cnt":                "fin_flag_count",
    "syn_flag_cnt":                "syn_flag_count",
    "rst_flag_cnt":                "rst_flag_count",
    "psh_flag_cnt":                "psh_flag_count",
    "ack_flag_cnt":                "ack_flag_count",
    "urg_flag_cnt":                "urg_flag_count",
    "cwe_flag_count":              "cwe_flag_count",
    "ece_flag_cnt":                "ece_flag_count",

    # Header lengths
    "fwd_header_len":              "fwd_header_length",
    "bwd_header_len":              "bwd_header_length",

    # Packet rates
    "fwd_pkts_s":                  "fwd_packets_s",
    "bwd_pkts_s":                  "bwd_packets_s",

    # Packet length summary
    "pkt_len_min":                 "min_packet_length",
    "pkt_len_max":                 "max_packet_length",
    "pkt_len_mean":                "packet_length_mean",
    "pkt_len_std":                 "packet_length_std",
    "pkt_len_var":                 "packet_length_variance",

    # Ratios and averages
    "down_up_ratio":               "down_up_ratio",
    "pkt_size_avg":                "average_packet_size",
    "fwd_seg_size_avg":            "avg_fwd_segment_size",
    "bwd_seg_size_avg":            "avg_bwd_segment_size",

    # Bulk rates
    "fwd_byts_b_avg":               "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg":               "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg":            "fwd_avg_bulk_rate",
    "bwd_byts_b_avg":               "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg":               "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg":            "bwd_avg_bulk_rate",

    # Subflows
    "subfl_fw_pk":                 "subflow_fwd_packets",
    "subfl_fw_byt":                "subflow_fwd_bytes",
    "subfl_bw_pkt":                "subflow_bwd_packets",
    "subfl_bw_byt":                "subflow_bwd_bytes",
    "subflow_fwd_pkts":            "subflow_fwd_packets",
    "subflow_fwd_byts":            "subflow_fwd_bytes",
    "subflow_bwd_pkts":            "subflow_bwd_packets",
    "subflow_bwd_byts":            "subflow_bwd_bytes",

    # Window sizes
    "init_fwd_win_byts":           "init_win_bytes_forward",
    "init_bwd_win_byts":           "init_win_bytes_backward",

    # Active / idle
    "atv_avg":                     "active_mean",
    "atv_std":                     "active_std",
    "atv_max":                     "active_max",
    "atv_min":                     "active_min",
    "idl_avg":                     "idle_mean",
    "idl_std":                     "idle_std",
    "idl_max":                     "idle_max",
    "idl_min":                     "idle_min",

    # Miscellaneous
    "fwd_act_data_pkts":            "act_data_pkt_fwd",
    "fwd_seg_size_min":             "min_seg_size_forward",
}

# =============================================================================
# UNSW-NB15 COLUMN RENAMING MAP
#
# Maps UNSW-NB15 feature names to their CIC2017 equivalents after
# normalise_columns() is applied.
# Only features with a meaningful conceptual equivalent are mapped.
# The rest are dropped via the common feature intersection.
#
# Reference:
#   UNSW-NB15 feature description:
#   https://research.unsw.edu.au/projects/unsw-nb15-dataset
# =============================================================================

UNSW_RENAME = {
    # Flow duration
    "dur":       "flow_duration",           # flow duration in seconds

    # Packet counts
    "spkts":     "total_fwd_packets",       # source -> dest packet count
    "dpkts":     "total_backward_packets",  # dest -> source packet count

    # Byte counts
    "sbytes":    "total_length_of_fwd_packets",   # source -> dest bytes
    "dbytes":    "total_length_of_bwd_packets",   # dest -> source bytes

    # Packet size means  (forward / backward)
    "smeansz":   "fwd_packet_length_mean",  # mean packet size src->dst
    "dmeansz":   "bwd_packet_length_mean",  # mean packet size dst->src

    # Inter-packet timing
    "sinpkt":    "fwd_iat_mean",            # mean inter-packet time src->dst
    "dinpkt":    "bwd_iat_mean",            # mean inter-packet time dst->src

    # Jitter  (maps to IAT std -- closest equivalent)
    "sjit":      "fwd_iat_std",             # src->dst jitter
    "djit":      "bwd_iat_std",             # dst->src jitter

    # Load (bits/sec -- closest to flow bytes/s)
    "sload":     "flow_bytes_s",            # src->dst bits per second
    "dload":     "bwd_packets_s",           # dst->src bits per second

    # TTL values  (no direct CIC equivalent -- map to closest flag features)
    # Excluded -- no meaningful CIC2017 equivalent

    # TCP window sizes
    "swin":      "init_win_bytes_forward",  # src TCP window size
    "dwin":      "init_win_bytes_backward", # dst TCP window size

    # TCP round trip / handshake timing
    "tcprtt":    "flow_iat_std",            # TCP round trip time (approx)
    "synack":    "fwd_iat_min",             # SYN to SYN-ACK time
    "ackdat":    "bwd_iat_min",             # SYN-ACK to ACK time

    # Loss counts  (map to flag counts -- closest equivalent)
    "sloss":     "fwd_packets_s",           # src retransmitted/dropped pkts
    "dloss":     "bwd_packet_length_std",   # dst retransmitted/dropped pkts (approx)

    # Port
    "dsport":    "destination_port",        # destination port

    # Connection counts (ct_ features -- statistical connection counters)
    "ct_srv_src":       "subflow_fwd_packets",
    "ct_srv_dst":       "subflow_bwd_packets",
    "ct_dst_ltm":       "subflow_fwd_bytes",
    "ct_src_ltm":       "subflow_bwd_bytes",
    "ct_src_dport_ltm": "fwd_avg_bytes_bulk",
    "ct_dst_sport_ltm": "bwd_avg_bytes_bulk",
    "ct_dst_src_ltm":   "fwd_avg_packets_bulk",

    # Binary flags
    "is_sm_ips_ports":  "syn_flag_count",
    "is_ftp_login":     "fin_flag_count",
    "ct_ftp_cmd":       "psh_flag_count",
}
