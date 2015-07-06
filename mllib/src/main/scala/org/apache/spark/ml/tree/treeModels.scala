/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree

import scala.collection.mutable
import org.apache.spark.ml.tree.LeafNode

/**
 * Abstraction for Decision Tree models.
 *
 * TODO: Add support for predicting probabilities and raw predictions  SPARK-3727
 */
private[ml] trait DecisionTreeModel {

  /** Root of the decision tree */
  def rootNode: Node

  /** Number of nodes in tree, including leaf nodes. */
  def numNodes: Int = {
    1 + rootNode.numDescendants
  }

  /**
   * Depth of the tree.
   * E.g.: Depth 0 means 1 leaf node.  Depth 1 means 1 internal node and 2 leaf nodes.
   */
  lazy val depth: Int = {
    rootNode.subtreeDepth
  }

  /** Summary of the model */
  override def toString: String = {
    // Implementing classes should generally override this method to be more descriptive.
    s"DecisionTreeModel of depth $depth with $numNodes nodes"
  }

  /** Full description of model */
  def toDebugString: String = {
    val header = toString + "\n"
    header + rootNode.subtreeToString(2)
  }

  lazy val variableImportance: Map[Int, Double] = {
    VariableImportance.computeVariableImportance(rootNode)
  }
}

/**
 * Abstraction for models which are ensembles of decision trees
 *
 * TODO: Add support for predicting probabilities and raw predictions  SPARK-3727
 */
private[ml] trait TreeEnsembleModel {

  // Note: We use getTrees since subclasses of TreeEnsembleModel will store subclasses of
  //       DecisionTreeModel.

  /** Trees in this ensemble. Warning: These have null parent Estimators. */
  def trees: Array[DecisionTreeModel]

  /** Weights for each tree, zippable with [[trees]] */
  def treeWeights: Array[Double]

  /** Summary of the model */
  override def toString: String = {
    // Implementing classes should generally override this method to be more descriptive.
    s"TreeEnsembleModel with $numTrees trees"
  }

  /** Full description of model */
  def toDebugString: String = {
    val header = toString + "\n"
    header + trees.zip(treeWeights).zipWithIndex.map { case ((tree, weight), treeIndex) =>
      s"  Tree $treeIndex (weight $weight):\n" + tree.rootNode.subtreeToString(4)
    }.fold("")(_ + _)
  }

  /** Number of trees in ensemble */
  val numTrees: Int = trees.length

  /** Total number of nodes, summed over all trees in the ensemble. */
  lazy val totalNumNodes: Int = trees.map(_.numNodes).sum

  lazy val variableImportance: Map[Int, Double] = {
    VariableImportance.computeVariableImportance(trees.toSeq.map(_.rootNode):_*)
  }
}


private[ml] object VariableImportance {
  /**
   * Computes variable importance by summing up impurity gains for each feature and normalizing them
   * over partition size and trees.
   * @param rootNodes Root nodes for each tree in the ensemble.
   * @return A mapping from feature index to importance value. If a feature index is not present its
   *         importance is zero.
   */
  def computeVariableImportance(rootNodes: Node*): Map[Int, Double] = {

    def recVarImp(node: Node, varImp: mutable.HashMap[Int, Double]): mutable.HashMap[Int, Double] = {
      node match {
        case terminal:LeafNode => varImp  // stop criterion
        case nonTerminal:InternalNode => {
          varImp(nonTerminal.split.featureIndex) += nonTerminal.gain
          varImp
        }
      }
    }

    def varImp(root: Node): mutable.HashMap[Int, Double] = {
      val m = new mutable.HashMap[Int, Double]()
      recVarImp(root, m)
    }

    def mergeMap(a: Map[Int, Double], b: Map[Int, Double]): Map[Int, Double] = {
      val o = (a.keySet ++ b.keySet).map(key => (key, a.getOrElse(key, 0.0) + b.getOrElse(key, 0.0)))
      o.toMap
    }
    val importance: Map[Int, Double] = rootNodes.asParIterable.map(varImp).reduce(mergeMap)
    if (rootNodes.length == 1)
      importance
    else {
      // divide by number of trees for ensembles
      importance.map(tpl => (tpl._1, tpl._2 / rootNodes.length)).toMap
    }
  }
}