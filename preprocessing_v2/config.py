"""
preprocessing_v2/config.py
===========================
Configuration for the v2 pipeline.

Key differences from v1:
  - Training: combined benign from CIC2017 + CIC2018 + UNSW (80% each)
  - Internal validation: held-out 20% from each training dataset
  - Generalisation test: BoT-IoT (never seen during training)
  - Separate artefact directory: artefacts_v2/
  - v1 files are untouched
"""

import os

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR      = r"C:/MLProject/zero_day_project"
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw")
ARTEFACT_DIR  = os.path.join(BASE_DIR, "artefacts_v2")

DATASET_PATHS = {
    "cic2017": os.path.join(DATA_DIR, "CIC2017"),
    "cic2018": os.path.join(DATA_DIR, "CIC2018"),
    "unsw":    os.path.join(DATA_DIR, "UNSW"),
    "botiot":  os.path.join(DATA_DIR, "bot_iot"),
}

UNSW_DATA_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

BOTIOT_DATA_FILES = [
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv",
]

# =============================================================================
# SPLIT SETTINGS
#
# v2 design (supervisor):
#   - 80% of each dataset's benign traffic  → combined training set
#   - 20% of each dataset's benign traffic  → internal validation
#   - BoT-IoT (all)                         → unseen generalisation test
# =============================================================================

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.20

# Max rows sampled from large datasets to keep memory manageable
MAX_TRAIN_ROWS_PER_DS = 600_000   # benign rows per dataset for training
MAX_VAL_ROWS_PER_DS   = 150_000   # benign rows per dataset for validation
MAX_BOTIOT_ROWS       = 500_000   # total BoT-IoT rows (benign + attack)

# =============================================================================
# FEATURE SELECTION
# =============================================================================

CORRELATION_THRESHOLD = 0.85

# =============================================================================
# UNSW COLUMN NAMES  (no header in raw files)
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
# NON-NUMERIC COLUMNS TO DROP
# =============================================================================

NON_NUMERIC_COLS = [
    "srcip", "dstip", "proto", "state", "service", "attack_cat",
    "flow_id", "source_ip", "destination_ip", "timestamp",
    # BoT-IoT specific non-numeric
    "flgs", "saddr", "daddr", "category", "subcategory",
]

# =============================================================================
# LOG TRANSFORM COLUMNS
# =============================================================================

LOG_TRANSFORM_COLS = [
    "flow_bytes_s", "fwd_bytes_s", "bwd_bytes_s",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "total_fwd_packets", "total_backward_packets",
    "sbytes", "dbytes",
    "fwd_packet_length_max", "fwd_packet_length_mean",
    "bwd_packet_length_max", "bwd_packet_length_mean",
    "average_packet_size", "smeansz", "dmeansz",
    "flow_duration", "flow_iat_mean", "flow_iat_max",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_max",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_max",
    "dur", "sinpkt", "dinpkt", "flow_packets_s",
    "sload", "dload",
    # BoT-IoT equivalents (before rename)
    "bytes", "pkts", "rate", "srate", "drate",
    "spkts", "dpkts",
]

# =============================================================================
# CIC2018 RENAME MAP  (identical to v1)
# =============================================================================

CIC2018_RENAME = {
    "dst_port":            "destination_port",
    "tot_fwd_pkts":        "total_fwd_packets",
    "tot_bwd_pkts":        "total_backward_packets",
    "totlen_fwd_pkts":     "total_length_of_fwd_packets",
    "totlen_bwd_pkts":     "total_length_of_bwd_packets",
    "fwd_pkt_len_max":     "fwd_packet_length_max",
    "fwd_pkt_len_min":     "fwd_packet_length_min",
    "fwd_pkt_len_mean":    "fwd_packet_length_mean",
    "fwd_pkt_len_std":     "fwd_packet_length_std",
    "bwd_pkt_len_max":     "bwd_packet_length_max",
    "bwd_pkt_len_min":     "bwd_packet_length_min",
    "bwd_pkt_len_mean":    "bwd_packet_length_mean",
    "bwd_pkt_len_std":     "bwd_packet_length_std",
    "flow_byts_s":         "flow_bytes_s",
    "flow_pkts_s":         "flow_packets_s",
    "flow_iat_mean":       "flow_iat_mean",
    "flow_iat_std":        "flow_iat_std",
    "flow_iat_max":        "flow_iat_max",
    "flow_iat_min":        "flow_iat_min",
    "fwd_iat_tot":         "fwd_iat_total",
    "fwd_iat_mean":        "fwd_iat_mean",
    "fwd_iat_std":         "fwd_iat_std",
    "fwd_iat_max":         "fwd_iat_max",
    "fwd_iat_min":         "fwd_iat_min",
    "bwd_iat_tot":         "bwd_iat_total",
    "bwd_iat_mean":        "bwd_iat_mean",
    "bwd_iat_std":         "bwd_iat_std",
    "bwd_iat_max":         "bwd_iat_max",
    "bwd_iat_min":         "bwd_iat_min",
    "fwd_header_len":      "fwd_header_length",
    "bwd_header_len":      "bwd_header_length",
    "fwd_pkts_s":          "fwd_packets_s",
    "bwd_pkts_s":          "bwd_packets_s",
    "pkt_len_min":         "min_packet_length",
    "pkt_len_max":         "max_packet_length",
    "pkt_len_mean":        "packet_length_mean",
    "pkt_len_std":         "packet_length_std",
    "pkt_len_var":         "packet_length_variance",
    "down_up_ratio":       "down_up_ratio",
    "pkt_size_avg":        "average_packet_size",
    "fwd_seg_size_avg":    "avg_fwd_segment_size",
    "bwd_seg_size_avg":    "avg_bwd_segment_size",
    "fwd_byts_b_avg":      "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg":      "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg":    "fwd_avg_bulk_rate",
    "bwd_byts_b_avg":      "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg":      "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg":    "bwd_avg_bulk_rate",
    "subfl_fw_pk":         "subflow_fwd_packets",
    "subfl_fw_byt":        "subflow_fwd_bytes",
    "subfl_bw_pkt":        "subflow_bwd_packets",
    "subfl_bw_byt":        "subflow_bwd_bytes",
    "subflow_fwd_pkts":    "subflow_fwd_packets",
    "subflow_fwd_byts":    "subflow_fwd_bytes",
    "subflow_bwd_pkts":    "subflow_bwd_packets",
    "subflow_bwd_byts":    "subflow_bwd_bytes",
    "init_fwd_win_byts":   "init_win_bytes_forward",
    "init_bwd_win_byts":   "init_win_bytes_backward",
    "atv_avg":             "active_mean",
    "atv_std":             "active_std",
    "atv_max":             "active_max",
    "atv_min":             "active_min",
    "idl_avg":             "idle_mean",
    "idl_std":             "idle_std",
    "idl_max":             "idle_max",
    "idl_min":             "idle_min",
    "fwd_act_data_pkts":   "act_data_pkt_fwd",
    "fwd_seg_size_min":    "min_seg_size_forward",
}

# =============================================================================
# UNSW RENAME MAP  (identical to v1)
# =============================================================================

UNSW_RENAME = {
    "dur":              "flow_duration",
    "spkts":            "total_fwd_packets",
    "dpkts":            "total_backward_packets",
    "sbytes":           "total_length_of_fwd_packets",
    "dbytes":           "total_length_of_bwd_packets",
    "smeansz":          "fwd_packet_length_mean",
    "dmeansz":          "bwd_packet_length_mean",
    "sinpkt":           "fwd_iat_mean",
    "dinpkt":           "bwd_iat_mean",
    "sjit":             "fwd_iat_std",
    "djit":             "bwd_iat_std",
    "sload":            "flow_bytes_s",
    "dload":            "bwd_packets_s",
    "swin":             "init_win_bytes_forward",
    "dwin":             "init_win_bytes_backward",
    "tcprtt":           "flow_iat_std",
    "synack":           "fwd_iat_min",
    "ackdat":           "bwd_iat_min",
    "sloss":            "fwd_packets_s",
    "dloss":            "bwd_packet_length_std",
    "dsport":           "destination_port",
    "ct_srv_src":       "subflow_fwd_packets",
    "ct_srv_dst":       "subflow_bwd_packets",
    "ct_dst_ltm":       "subflow_fwd_bytes",
    "ct_src_ltm":       "subflow_bwd_bytes",
    "ct_src_dport_ltm": "fwd_avg_bytes_bulk",
    "ct_dst_sport_ltm": "bwd_avg_bytes_bulk",
    "ct_dst_src_ltm":   "fwd_avg_packets_bulk",
    "is_sm_ips_ports":  "syn_flag_count",
    "is_ftp_login":     "fin_flag_count",
    "ct_ftp_cmd":       "psh_flag_count",
}

# =============================================================================
# BOT-IOT RENAME MAP
#
# Maps BoT-IoT feature names (post normalise_columns) to CIC2017 equivalents.
# BoT-IoT is a flow-based dataset with similar concepts but different names.
#
# Reference:
#   Koroniotis et al. (2019) "Towards the Development of Realistic Botnet
#   Dataset in the Internet of Things for Network Forensic Analytics"
#   https://doi.org/10.1016/j.future.2019.05.041
# =============================================================================

BOTIOT_RENAME = {
    # Flow duration
    "dur":          "flow_duration",        # flow duration in seconds

    # Packet counts (src=fwd, dst=bwd)
    "spkts":        "total_fwd_packets",    # source packets
    "dpkts":        "total_backward_packets",  # destination packets
    "pkts":         "average_packet_size",  # total pkts -> use as avg proxy

    # Byte counts
    "sbytes":       "total_length_of_fwd_packets",
    "dbytes":       "total_length_of_bwd_packets",
    "bytes":        "max_packet_length",    # total bytes -> proxy

    # Flow rates
    "rate":         "flow_packets_s",       # total packet rate
    "srate":        "fwd_packets_s",        # source packet rate
    "drate":        "bwd_packets_s",        # destination packet rate

    # Packet size statistics
    "mean":         "packet_length_mean",   # mean packet size
    "stddev":       "packet_length_std",    # std of packet size
    "sum":          "packet_length_variance",  # sum -> proxy for variance
    "min":          "min_packet_length",    # min packet size
    "max":          "fwd_packet_length_max",  # max packet size

    # Port
    "dport":        "destination_port",     # destination port

    # Protocol number (numeric encoding)
    "proto_number": "fwd_header_length",    # protocol -> proxy

    # State number (connection state encoding)
    "state_number": "bwd_header_length",    # state -> proxy

    # Connection tracking counters (statistical features)
    "tnbpsrcip":    "subflow_fwd_packets",  # total pkts by src IP
    "tnbpdstip":    "subflow_bwd_packets",  # total pkts by dst IP
    "tnp_psrcip":   "subflow_fwd_bytes",    # total pkts per src IP protocol
    "tnp_pdstip":   "subflow_bwd_bytes",    # total pkts per dst IP protocol
    "tnp_perproto": "fwd_avg_bytes_bulk",   # total pkts per protocol
    "tnp_per_dport":"bwd_avg_bytes_bulk",   # total pkts per dest port
    "ar_p_proto_p_srcip":  "fwd_avg_packets_bulk",  # arrival rate src IP
    "ar_p_proto_p_dstip":  "bwd_avg_packets_bulk",  # arrival rate dst IP
    "n_in_conn_p_dstip":   "fwd_avg_bulk_rate",     # inbound conns per dst
    "n_in_conn_p_srcip":   "bwd_avg_bulk_rate",     # inbound conns per src
    "ar_p_proto_p_sport":  "init_win_bytes_forward", # arrival rate src port
    "ar_p_proto_p_dport":  "init_win_bytes_backward",# arrival rate dst port
    "pkts_p_state_p_protocol_p_destip": "syn_flag_count",
    "pkts_p_state_p_protocol_p_srcip":  "fin_flag_count",

    # Label column
    "attack":       "label",
}

# Label column name per dataset (after loading, before rename)
LABEL_COLS = {
    "cic2017": "label",
    "cic2018": "label",
    "unsw":    "label",
    "botiot":  "attack",
}
