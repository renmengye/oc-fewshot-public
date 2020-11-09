"""Build models.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.model_factory import get_model
from fewshot.models.model_factory import get_module


def build_backbone(config):
  """Builds a backbone network."""
  if config.backbone_class in [
      'c4_backbone', 'c4_double_backbone', 'c4_gn_backbone',
      'c4_plus_fc_backbone'
  ]:
    bb_config = config.c4_config
  elif config.backbone_class in [
      'resnet_backbone', 'resnet_gn_backbone', 'resnet_snail_backbone',
      'resnet_12_backbone'
  ]:
    bb_config = config.resnet_config
  else:
    raise ValueError('Unknown backbone class {}'.format(config.backbone_class))
  bb = get_module(config.backbone_class, bb_config)
  if config.roi_pooling_backbone:
    bb = get_module('roi_pooling_backbone', bb_config, bb)
  return bb


def build_pretrain_net(config, backbone=None):
  """Builds a regular classification network for pretraining."""
  if backbone is None:
    backbone = build_backbone(config)
  model = get_model("pretrain_net", config, backbone)
  return model


def build_memory_module(config, backbone):
  """Builds a memory module."""
  D = backbone.get_output_dimension()[0]
  K = config.num_classes + 1
  if config.model_class in ['lstm_net', 'lstm_sigmoid_net']:
    if config.add_stage_id:
      inp_dim = D + 2 * K
    else:
      inp_dim = D + K
  elif config.model_class in [
      'online_proto_net', 'online_proto_sigmoid_net', 'proto_mem_net',
      'proto_mem_sigmoid_net', 'oml_sigmoid_net', 'oml_trunc_sigmoid_net',
      'online_example_net', 'mixture_proto_net', 'online_classifier_net'
  ]:
    inp_dim = D
  elif config.model_class in ['cpm_net', 'proto_plus_rnn_net']:
    inp_dim = D
  else:
    raise ValueError('Unknown model class {}'.format(config.model_class))

  if config.memory_class in [
      'min_dist_proto_memory', 'ssl_min_dist_proto_memory',
      'ssl_min_dist_forget_proto_memory'
  ]:
    name = 'proto_memory'
    memory = get_module(
        config.memory_class,
        name,
        inp_dim,
        config.memory_net_config.radius_init,
        max_classes=config.memory_net_config.max_classes,
        fix_unknown=config.fix_unknown,
        unknown_id=config.num_classes if config.fix_unknown else None,
        similarity=config.memory_net_config.similarity,
        radius_init_write=config.memory_net_config.radius_init_write,
        use_ssl_beta_gamma_write=config.hybrid_config.use_ssl_beta_gamma_write,
        unknown_logits=config.memory_net_config.unknown_logits,
        temp_init=config.memory_net_config.temp_init,
        dtype=tf.float32)

  elif config.memory_class in [
      'online_matchingnet_memory', 'online_imp_memory',
      'online_mixture_memory', 'matchingnet_memory', 'online_ocsvm',
      'online_ovrsvm', 'online_lr'
  ]:
    name = 'imp'
    memory = get_module(
        config.memory_class,
        name,
        inp_dim,
        max_items=config.memory_net_config.max_items,
        max_classes=config.memory_net_config.max_classes,
        unknown_id=config.num_classes,
        log_sigma_init=config.memory_net_config.log_sigma_init,
        log_lambda_init=config.memory_net_config.log_lambda_init,
        radius_init=config.memory_net_config.radius_init,
        similarity=config.memory_net_config.similarity,
        dtype=tf.float32)
  elif config.memory_class in ['lstm']:
    name = 'lstm'
    memory = get_module(
        config.memory_class,
        name,
        inp_dim,
        config.lstm_config.hidden_dim,
        layernorm=config.lstm_config.layernorm,
        dtype=tf.float32)
  elif config.memory_class in ['stack_lstm']:
    name = config.memory_class
    memory = get_module(
        config.memory_class,
        name,
        inp_dim,
        config.lstm_config.hidden_dim,
        config.lstm_config.nstack,
        layernorm=config.lstm_config.layernorm,
        dtype=tf.float32)
  elif config.memory_class in ['dnc']:
    name = config.memory_class
    memory = get_module(
        config.memory_class,
        name,
        inp_dim,
        config.mann_config,
        # config.mann_config.memory_dim[0],
        # config.mann_config.controller_dim[0],
        # config.mann_config.num_slots[0],
        # config.mann_config.num_reads[0],
        # config.mann_config.num_writes[0],
        # controller_type=config.mann_config.controller_type,
        # memory_layernorm=config.mann_config.memory_layernorm,
        # controller_layernorm=config.mann_config.controller_layernorm,
        # controller_nstack=config.mann_config.controller_nstack,
        # similarity_type=config.mann_config.similarity_type,
        dtype=tf.float32)
  elif config.memory_class in [
      'dnc_writehead_v2',
      'dnc_writeheadfeed2',
  ]:
    label_dim = K
    name = config.memory_class
    memory = get_module(
        config.memory_class,
        name,
        inp_dim,
        label_dim,
        config.mann_config,
        # config.mann_config.memory_dim[0],
        # config.mann_config.controller_dim[0],
        # config.mann_config.num_slots[0],
        # config.mann_config.num_reads[0],
        # config.mann_config.num_writes[0],
        # controller_type=config.mann_config.controller_type,
        # memory_layernorm=config.mann_config.memory_layernorm,
        # controller_layernorm=config.mann_config.controller_layernorm,
        # controller_nstack=config.mann_config.controller_nstack,
        # similarity_type=config.mann_config.similarity_type,
        dtype=tf.float32)

  elif config.memory_class in ['proto_plus_rnn_v4', 'cpm']:
    memory_class = config.sub_memory_class
    memory_class2 = config.sub_memory_class2
    if memory_class.startswith('dnc'):
      rnn_memory = get_module(
          memory_class,
          'dnc',
          inp_dim,
          config.mann_config.memory_dim,
          config.mann_config.controller_dim,
          config.mann_config.num_slots,
          config.mann_config.num_reads,
          config.mann_config.num_writes,
          controller_type=config.mann_config.controller_type,
          memory_layernorm=config.mann_config.memory_layernorm,
          controller_layernorm=config.mann_config.controller_layernorm,
          controller_nstack=config.mann_config.controller_nstack,
          similarity_type=config.mann_config.similarity_type,
          dtype=tf.float32)
    elif memory_class in ['stack_lstm']:
      rnn_memory = get_module(
          memory_class,
          "stack_lstm",
          inp_dim,
          config.lstm_config.hidden_dim,
          config.lstm_config.nstack,
          layernorm=config.lstm_config.layernorm,
          dtype=tf.float32)
    elif memory_class in ['lstm']:
      rnn_memory = get_module(
          memory_class,
          "lstm",
          inp_dim,
          config.lstm_config.hidden_dim,
          layernorm=config.lstm_config.layernorm,
          dtype=tf.float32)
    proto_memory = get_module(
        memory_class2,
        'proto_memory',
        inp_dim,
        config.memory_net_config.radius_init,
        max_classes=config.memory_net_config.max_classes,
        fix_unknown=config.fix_unknown,
        unknown_id=config.num_classes if config.fix_unknown else None,
        similarity=config.memory_net_config.similarity,
        static_beta_gamma=not config.hybrid_config.use_pred_beta_gamma,
        unknown_logits=config.memory_net_config.unknown_logits,
        temp_init=config.memory_net_config.temp_init,
        dtype=tf.float32)
    proto_plus_rnn = get_module(
        config.memory_class,
        'proto_plus_rnn',
        proto_memory,
        rnn_memory,
        readout_type=config.mann_config.readout_type,
        use_pred_beta_gamma=config.hybrid_config.use_pred_beta_gamma,
        use_feature_fuse=config.hybrid_config.use_feature_fuse,
        use_feature_fuse_gate=config.hybrid_config.use_feature_fuse_gate,
        use_feature_scaling=config.hybrid_config.use_feature_scaling,
        use_feature_memory_only=config.hybrid_config.use_feature_memory_only,
        skip_unk_memory_update=config.hybrid_config.skip_unk_memory_update,
        dtype=tf.float32)
    return proto_plus_rnn

  elif config.memory_class in ['proto_plus_rnn_ssl_v4', 'cpm_ssl']:
    memory_class = config.sub_memory_class
    memory_class2 = config.sub_memory_class2
    if memory_class.startswith('dnc'):
      rnn_memory = get_module(
          memory_class,
          'dnc',
          inp_dim,
          config.mann_config.memory_dim,
          config.mann_config.controller_dim,
          config.mann_config.num_slots,
          config.mann_config.num_reads,
          config.mann_config.num_writes,
          controller_type=config.mann_config.controller_type,
          memory_layernorm=config.mann_config.memory_layernorm,
          controller_layernorm=config.mann_config.controller_layernorm,
          controller_nstack=config.mann_config.controller_nstack,
          similarity_type=config.mann_config.similarity_type,
          dtype=tf.float32)
    elif memory_class in ['stack_lstm']:
      rnn_memory = get_module(
          memory_class,
          "stack_lstm",
          inp_dim,
          config.lstm_config.hidden_dim,
          config.lstm_config.nstack,
          layernorm=config.lstm_config.layernorm,
          dtype=tf.float32)
    elif memory_class in ['lstm']:
      rnn_memory = get_module(
          memory_class,
          "lstm",
          inp_dim,
          config.lstm_config.hidden_dim,
          layernorm=config.lstm_config.layernorm,
          dtype=tf.float32)
    proto_memory = get_module(
        memory_class2,
        'proto_memory',
        inp_dim,
        config.memory_net_config.radius_init,
        max_classes=config.memory_net_config.max_classes,
        fix_unknown=config.fix_unknown,
        unknown_id=config.num_classes if config.fix_unknown else None,
        similarity=config.memory_net_config.similarity,
        static_beta_gamma=not config.hybrid_config.use_pred_beta_gamma,
        unknown_logits=config.memory_net_config.unknown_logits,
        temp_init=config.memory_net_config.temp_init,
        dtype=tf.float32)
    proto_plus_rnn = get_module(
        config.memory_class,
        'proto_plus_rnn',
        proto_memory,
        rnn_memory,
        readout_type=config.mann_config.readout_type,
        use_pred_beta_gamma=config.hybrid_config.use_pred_beta_gamma,
        use_feature_fuse=config.hybrid_config.use_feature_fuse,
        use_feature_fuse_gate=config.hybrid_config.use_feature_fuse_gate,
        use_feature_scaling=config.hybrid_config.use_feature_scaling,
        use_feature_memory_only=config.hybrid_config.use_feature_memory_only,
        skip_unk_memory_update=config.hybrid_config.skip_unk_memory_update,
        use_ssl=config.hybrid_config.use_ssl,
        use_ssl_beta_gamma_write=config.hybrid_config.use_ssl_beta_gamma_write,
        use_ssl_temp=config.hybrid_config.use_ssl_temp,
        dtype=tf.float32)
    return proto_plus_rnn

  elif config.memory_class in ['oml']:
    oml = get_module(
        config.memory_class, 'oml', config.oml_config, dtype=tf.float32)
    return oml

  else:
    raise ValueError('Unknown memory class {}'.format(config.memory_class))
  return memory


def build_net(config, backbone=None, memory=None, distributed=False):
  """Build a memory based lifelong learning model.

  Args:
    config: Model config.
    backbone: Backbone network.
    memory: Memory network.
  """
  if backbone is None:
    backbone = build_backbone(config)
  if memory is None:
    memory = build_memory_module(config, backbone)
  model = get_model(
      config.model_class, config, backbone, memory, distributed=distributed)
  return model
