# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import modeling
import optimization
import tokenization
import tensorflow as tf
tf.enable_eager_execution()

import tf_metrics
import pickle

from itertools import chain

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,  # prevent accent marks from being removed
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

# flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

# flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

# flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids):
            #    is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    # self.is_real_example = is_real_example

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file, eos='。'):
    """Reads a IOB data."""
    # with tf.gfile.Open(input_file, "r") as f:
    with open(input_file) as f:
      lines = []
      words = []
      labels = []
      for line in f:
        contents = line.strip()
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        if contents.startswith("-DOCSTART-"):
          words.append('')
          continue
        if len(words) > 0 and len(contents) == 0 and words[-1] == eos:
          l = ' '.join([label for label in labels if len(label) > 0])
          w = ' '.join([word for word in words if len(word) > 0])
          lines.append([l, w])
          words = []
          labels = []
          continue
        words.append(word)
        labels.append(label)
      return lines


class NerProcessor(DataProcessor):

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
    return [
 'I-Academic',
 'B-Academic',
 'I-Age',
 'B-Age',
 'I-Aircraft',
 'B-Aircraft',
 'B-Airport',
 'I-Airport',
 'B-Amphibia',
 'I-Amusement_Park',
 'B-Amusement_Park',
 'B-Animal_Disease',
 'I-Animal_Disease',
 'B-Animal_Part',
 'I-Animal_Part',
 'B-Archaeological_Place_Other',
 'I-Archaeological_Place_Other',
 'I-Art_Other',
 'B-Art_Other',
 'I-Astral_Body_Other',
 'B-Astral_Body_Other',
 'I-Award',
 'B-Award',
 'I-Bay',
 'B-Bay',
 'B-Bird',
 'I-Bird',
 'B-Book',
 'I-Book',
 'I-Bridge',
 'B-Bridge',
 'B-Broadcast_Program',
 'I-Broadcast_Program',
 'B-Cabinet',
 'I-Cabinet',
 'I-Calorie',
 'B-Calorie',
 'B-Canal',
 'I-Canal',
 'B-Car',
 'I-Car',
 'B-Car_Stop',
 'I-Car_Stop',
 'B-Character',
 'I-Character',
 'B-City',
 'I-City',
 'I-Class',
 'B-Class',
 'I-Clothing',
 'B-Clothing',
 'B-Color_Other',
 'I-Color_Other',
 'I-Company',
 'B-Company',
 'B-Company_Group',
 'I-Company_Group',
 'I-Compound',
 'B-Compound',
 'B-Conference',
 'I-Conference',
 'B-Constellation',
 'I-Constellation',
 'I-Continental_Region',
 'B-Continental_Region',
 'B-Corporation_Other',
 'I-Corporation_Other',
 'I-Country',
 'B-Country',
 'I-Countx_Other',
 'B-Countx_Other',
 'B-County',
 'I-County',
 'B-Culture',
 'I-Culture',
 'B-Currency',
 'I-Currency',
 'B-Date',
 'I-Date',
 'I-Day_Of_Week',
 'B-Day_Of_Week',
 'B-Decoration',
 'I-Decoration',
 'I-Disease_Other',
 'B-Disease_Other',
 'B-Dish',
 'I-Dish',
 'B-Doctrine_Method_Other',
 'I-Doctrine_Method_Other',
 'B-Domestic_Region',
 'I-Domestic_Region',
 'I-Drug',
 'B-Drug',
 'B-Earthquake',
 'I-Earthquake',
 'I-Element',
 'B-Element',
 'B-Email',
 'I-Email',
 'B-Era',
 'I-Era',
 'I-Ethnic_Group_Other',
 'B-Ethnic_Group_Other',
 'B-Event_Other',
 'I-Event_Other',
 'I-Facility_Other',
 'B-Facility_Other',
 'B-Facility_Part',
 'I-Facility_Part',
 'I-Family',
 'B-Family',
 'B-Fish',
 'I-Fish',
 'B-Flora',
 'I-Flora',
 'I-Flora_Part',
 'B-Flora_Part',
 'I-Food',
 'B-Food',
 'I-Food_Other',
 'B-Food_Other',
 'I-Freguency',
 'B-Freguency',
 'I-Frequency',
 'B-Frequency',
 'I-Fungus',
 'B-Fungus',
 'B-GOE_Other',
 'I-GOE_Other',
 'B-GPE_Other',
 'I-GPE_Other',
 'B-Game',
 'I-Game',
 'I-Geological_Region_Other',
 'B-Geological_Region_Other',
 'I-God',
 'B-God',
 'B-Government',
 'I-Government',
 'B-ID_Number',
 'I-ID_Number',
 'B-Incident_Other',
 'I-Incident_Other',
 'I-Insect',
 'B-Insect',
 'B-Intensity',
 'I-Intensity',
 'I-International_Organization',
 'B-International_Organization',
 'B-Island',
 'I-Island',
 'B-Lake',
 'I-Lake',
 'I-Language_Other',
 'B-Language_Other',
 'B-Latitude_Longtitude',
 'I-Latitude_Longtitude',
 'B-Law',
 'I-Law',
 'I-Line_Other',
 'B-Line_Other',
 'I-Living_Thing_Other',
 'B-Living_Thing_Other',
 'I-Living_Thing_Part_Other',
 'B-Living_Thing_Part_Other',
 'I-Location_Other',
 'B-Location_Other',
 'I-Magazine',
 'B-Magazine',
 'I-Mammal',
 'B-Mammal',
 'I-Market',
 'B-Market',
 'I-Material',
 'B-Material',
 'I-Measurement_Other',
 'B-Measurement_Other',
 'I-Military',
 'B-Military',
 'B-Mineral',
 'I-Mineral',
 'B-Mollusc_Arthropod',
 'I-Mollusc_Arthropod',
 'I-Money',
 'B-Money',
 'B-Money_Form',
 'I-Money_Form',
 'B-Mountain',
 'I-Mountain',
 'B-Movement',
 'I-Movement',
 'I-Movie',
 'B-Movie',
 'I-Multiplication',
 'B-Multiplication',
 'I-Museum',
 'B-Museum',
 'B-Music',
 'I-Music',
 'B-N_Animal',
 'I-N_Animal',
 'I-N_Country',
 'B-N_Country',
 'B-N_Event',
 'I-N_Event',
 'B-N_Facility',
 'I-N_Facility',
 'I-N_Flora',
 'B-N_Flora',
 'B-N_Location_Other',
 'I-N_Location_Other',
 'I-N_Natural_Object_Other',
 'B-N_Natural_Object_Other',
 'I-N_Organization',
 'B-N_Organization',
 'I-N_Person',
 'B-N_Person',
 'I-N_Product',
 'B-N_Product',
 'I-Name_Other',
 'B-Name_Other',
 'B-National_Language',
 'I-National_Language',
 'I-Nationality',
 'B-Nationality',
 'I-Natural_Disaster',
 'B-Natural_Disaster',
 'B-Natural_Object_Other',
 'I-Natural_Object_Other',
 'B-Natural_Phenomenon_Other',
 'I-Natural_Phenomenon_Other',
 'I-Nature_Color',
 'B-Nature_Color',
 'I-Newspaper',
 'B-Newspaper',
 'B-Numex_Other',
 'I-Numex_Other',
 'B-Occasion_Other',
 'I-Occasion_Other',
 'B-Offence',
 'I-Offence',
 'B-Offense',
 'I-Offense',
 'I-Ordinal_Number',
 'B-Ordinal_Number',
 'I-Organization_Other',
 'B-Organization_Other',
 'B-Park',
 'I-Park',
 'B-Percent',
 'I-Percent',
 'I-Period_Day',
 'B-Period_Day',
 'I-Period_Month',
 'B-Period_Month',
 'I-Period_Time',
 'B-Period_Time',
 'B-Period_Week',
 'I-Period_Week',
 'B-Period_Year',
 'I-Period_Year',
 'I-Period_time',
 'B-Period_time',
 'I-Periodx_Other',
 'B-Periodx_Other',
 'I-Person',
 'B-Person',
 'I-Phone_Number',
 'B-Phone_Number',
 'B-Physical_Extent',
 'I-Physical_Extent',
 'I-Picture',
 'B-Picture',
 'I-Plan',
 'B-Plan',
 'B-Planet',
 'I-Planet',
 'B-Point',
 'I-Point',
 'I-Political_Organization_Other',
 'B-Political_Organization_Other',
 'B-Political_Party',
 'I-Political_Party',
 'I-Port',
 'B-Port',
 'B-Position_Vocation',
 'I-Position_Vocation',
 'B-Position_vocation',
 'I-Position_vocation',
 'B-Postal_Address',
 'I-Postal_Address',
 'B-Printing_Other',
 'I-Printing_Other',
 'I-Pro_Sports_Organization',
 'B-Pro_Sports_Organization',
 'B-Product_Other',
 'I-Product_Other',
 'I-Province',
 'B-Province',
 'B-Public_Institution',
 'I-Public_Institution',
 'I-Railroad',
 'B-Railroad',
 'B-Rank',
 'I-Rank',
 'I-Region_Other',
 'B-Region_Other',
 'B-Religion',
 'I-Religion',
 'I-Religious_Festival',
 'B-Religious_Festival',
 'B-Reptile',
 'I-Reptile',
 'I-Research_Institute',
 'B-Research_Institute',
 'B-River',
 'I-River',
 'I-Road',
 'B-Road',
 'I-Rule_Other',
 'B-Rule_Other',
 'B-School',
 'I-School',
 'I-School_Age',
 'B-School_Age',
 'I-Sea',
 'B-Sea',
 'I-Seismic_Intensity',
 'B-Seismic_Intensity',
 'I-Ship',
 'B-Ship',
 'B-Show',
 'I-Show',
 'B-Show_Organization',
 'I-Show_Organization',
 'B-Spa',
 'I-Spa',
 'I-Space',
 'B-Space',
 'I-Spaceship',
 'B-Spaceship',
 'B-Speed',
 'I-Speed',
 'B-Sport',
 'I-Sport',
 'I-Sports_Facility',
 'B-Sports_Facility',
 'I-Sports_League',
 'B-Sports_League',
 'I-Sports_Organization_Other',
 'B-Sports_Organization_Other',
 'B-Star',
 'I-Star',
 'I-Station',
 'B-Station',
 'B-Stock',
 'I-Stock',
 'B-Stock_Index',
 'I-Stock_Index',
 'B-Style',
 'I-Style',
 'I-Temperature',
 'B-Temperature',
 'B-Theater',
 'I-Theater',
 'I-Theory',
 'B-Theory',
 'B-Time',
 'I-Time',
 'B-Time_Top_Other',
 'I-Time_Top_Other',
 'B-Timex_Other',
 'I-Timex_Other',
 'B-Title_Other',
 'I-Title_Other',
 'I-Train',
 'B-Train',
 'B-Treaty',
 'I-Treaty',
 'I-Tumulus',
 'B-Tumulus',
 'I-Tunnel',
 'B-Tunnel',
 'B-URL',
 'I-URL',
 'B-Unit_Other',
 'I-Unit_Other',
 'B-Vehicle_Other',
 'I-Vehicle_Other',
 'B-Volume',
 'I-Volume',
 'I-War',
 'B-War',
 'I-Water_Route',
 'B-Water_Route',
 'B-Weapon',
 'I-Weapon',
 'B-Weight',
 'I-Weight',
 'B-Worship_Place',
 'I-Worship_Place',
 'B-Zoo',
 'I-Zoo',
            "O",
            "X", "[CLS]", "[SEP]"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    # set_type: train/dev/test
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text=text, label=label))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):

  """Converts a single `InputExample` into a single `InputFeatures`."""

  def write_tokens(tokens):
    if mode=="test":
      path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
      wf = open(path,'a')
      for token in tokens:
        if token!="[NULL]":
          wf.write(token+'\n')
      wf.close()

  label_map = {}
  for (i, label) in enumerate(label_list, 1):
    label_map[label] = i
  id2label_path = os.path.join(FLAGS.output_dir, "label2id.pkl")
  with open(id2label_path,'wb') as w:
    pickle.dump(label_map, w)

  textlist = example.text.split(' ')
  labellist = example.label.split(' ')
  tokens = []
  labels = []
  for i, word in enumerate(textlist):
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    label_1 = labellist[i]
    for m in range(len(token)):
      if m == 0:
        labels.append(label_1)
      else:
        labels.append("X")

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens) > max_seq_length - 2:
    tokens = tokens[0:(max_seq_length - 2)]
    labels = labels[0:(max_seq_length - 2)]

  ntokens = []
  segment_ids = []
  label_ids = []
  ntokens.append("[CLS]")
  segment_ids.append(0)
  # append("O") or append("[CLS]") not sure!
  label_ids.append(label_map["[CLS]"])
  for i, token in enumerate(tokens):
    ntokens.append(token)
    segment_ids.append(0)
    label_id = label_map[labels[i]] if labels[i] in label_map else label_map['O']
    label_ids.append(label_id)
  ntokens.append("[SEP]")
  segment_ids.append(0)
  # append("O") or append("[SEP]") not sure!
  label_ids.append(label_map["[SEP]"])

  input_ids = tokenizer.convert_tokens_to_ids(ntokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(0)
    ntokens.append("[NULL]")

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_ids)

  write_tokens(ntokens)
  return feature

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file,
    mode=None):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, mode)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a token-level classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_sequence_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    output_layer = tf.reshape(output_layer, [-1, hidden_size])

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    predict = tf.argmax(probabilities, axis=-1)

    return (loss, per_example_loss, logits, predict)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, predicts) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predicts,  # {"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


class SubwordWordConverter:

    # 18: 'X', 19: '[CLS]', 20: '[SEP]'
    # 'X'　は subword　-> word の復元に用いる
    ignore_label_ids = {0, 19, 20}
    # 0:[PAD] 1:[UNK] 2:[CLS] 3:[SEP] 4:[MASK]
    ignore_token_ids = {0, 2, 3, 4}
    
    def __init__(self, tokenizer, id2label):
        self.tokenizer = tokenizer
        self.id2label = id2label

    @staticmethod
    def convert_subword_to_word_by_label(subword_labels):
        # subword.startswith('##') == True だけがsubwordとは限らない
        words, labels = [], []
        for sw, lb in subword_labels:
            if lb == 'X':
                assert len(words) > 0
                prev = words[-1]
                words = words[:-1]
                word = prev + sw[2:]
            else:
                word = sw
                labels.append(lb)
            words.append(word)
        return words, labels

    @staticmethod
    def check_separator(inputs, labels):
        for i, l in zip(inputs, labels):
            if l == 19:
                if i != 2:
                    return False
            elif l == 20:
                if i != 3:
                    return False
        return True

    def convert_tokens_to_words(self, token_labels):
        token_labels = [(i, l) for i, l in token_labels if l not in self.ignore_label_ids]
        token_labels = [(i, l) for i, l in token_labels if i not in self.ignore_token_ids]
        token_ids, label_ids = zip(*token_labels)
        # subword　-> word の復元
        subwords = self.tokenizer.convert_ids_to_tokens(token_ids)
        labels = [self.id2label[i] for i in label_ids]
        words, labels = self.convert_subword_to_word_by_label(zip(subwords, labels))
        return words, labels

    def convert_labels_by_gold(self, labels_ids, labels_gold_ids):
        labels_ids = [l for l in labels_ids if l not in self.ignore_label_ids]
        labels_gold_ids = [l for l in labels_gold_ids if l not in self.ignore_label_ids]
        labels_ids = [l for l, lg in zip(labels_ids, labels_gold_ids) if lg != 18]
        return labels_ids

    def convert_tokens_to_words_gold(self, inputs, labels_ids, labels_ids_gold):
        # words, labels = self.convert_tokens_to_words(self, inputs, labels)
        # labels_gold = self.convert_labels_gold(labels_gold)
        words, labels_gold = self.convert_tokens_to_words(zip(inputs, labels_ids_gold))
        labels_ids = self.convert_labels_by_gold(labels_ids, labels_ids_gold)
        labels = [self.id2label[i] for i in labels_ids]
        return words, labels, labels_gold

    def convert_tokens_to_words_list(self, inputs_list, labels_list_pred, labels_list_gold):
      words_labels = [self.convert_tokens_to_words_gold(ins, lbs, lbs_g)
                      for ins, lbs, lbs_g in zip(inputs_list, labels_list_pred, labels_list_gold)
                      if self.check_separator(ins, lbs)]
      inputs_word, labels_word, labels_word_gold = zip(*words_labels)
      return inputs_word, labels_word, labels_word_gold

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  task_name = FLAGS.task_name.lower()

  processors = {
    "ner": NerProcessor
  }

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=None,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          )

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list) + 1,  # NOTE
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=0.,
      num_train_steps=None,
      num_warmup_steps=None,
      use_tpu=None,
      use_one_hot_embeddings=None)

  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=None,
      model_fn=model_fn,
      config=run_config,
    #   train_batch_size=FLAGS.train_batch_size,
    #   eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_predict:
    # read and export tokens
    token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
    id2label_path = os.path.join(FLAGS.output_dir, "label2id.pkl")
    with open(id2label_path,'rb') as rf:
      label2id = pickle.load(rf)
      id2label = {value:key for key,value in label2id.items()}
    if os.path.exists(token_path):
      os.remove(token_path)
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file, mode="test")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    # predict labels
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True)
    result = estimator.predict(input_fn=predict_input_fn)

    # obtain the processed tokens and outputs according to the original input words
    input_iter = predict_input_fn({'batch_size': 8})
    inputs_pred = [labels for input_batch in input_iter
                   for labels in input_batch['input_ids'].numpy()]
    input_iter = predict_input_fn({'batch_size': 8})
    labels_gold = [labels for input_batch in input_iter
                   for labels in input_batch['label_ids'].numpy()]
    assert len(inputs_pred) == len(labels_gold)
    swc = SubwordWordConverter(tokenizer, id2label)
    output_predict_file = os.path.join(FLAGS.output_dir, "token_label_pred.txt")
    # tokens_as_words, labels_per_word = swc.convert_tokens_to_words_list(inputs_pred, result)
    # with open(output_predict_file, 'w') as writer:
    #   tf.logging.info("***** Predict results *****")
    #   for i, (tokens, labels) in enumerate(zip(tokens_as_words, labels_per_word)):
    #     output_line = "\n".join(
    #         f'{token}\t{label}'
    #         for token, label in zip(tokens, labels)) + "\n\n"
    #     writer.write(output_line)
    tokens_list, labels_list, labels_gold_list = swc.convert_tokens_to_words_list(inputs_pred, result, labels_gold)
    with open(output_predict_file, 'w') as writer:
      tf.logging.info("***** Predict results *****")
      for i, (tokens, labels, labels_gold) in enumerate(zip(tokens_list, labels_list, labels_gold_list)):
        output_line = "\n".join(
            f'{token}\t{label}\t{label_gold}'
            for token, label, label_gold in zip(tokens, labels, labels_gold)) + "\n\n"
        writer.write(output_line)

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
