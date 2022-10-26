package server

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import ai.onnxruntime.OrtSession.SessionOptions

import scala.jdk.CollectionConverters._
import scala.util.Random

case class Model(path: String) {
  private val env = OrtEnvironment.getEnvironment
  val session: OrtSession = env.createSession(path, new SessionOptions())
  private val argNames = session.getInputNames.asScala.toSeq
  println(f"Model argnames: ${argNames.mkString(", ")}") // Currently contentId, size


  def map_in(values: Seq[Long]) = {
    argNames.zip(values).map(x => x._1 -> OnnxTensor.createTensor(env, scala.Array(x._2))).toMap.asJava
  }

  //TODO these things need to kinda be interfaces or contracts somehow.
  def map_out(output: OrtSession.Result) = {
    val scores = output.get(0).getValue.asInstanceOf[Array[Float]]
    val indices = output.get(1).getValue.asInstanceOf[Array[Long]]
    indices.zip(scores).map(x => x._1.toString -> x._2) // todo collapse option, only return relevant.
  }
  def predict_and_rank(values: Seq[Long]) = {
    map_out(session.run(map_in(values)))
  }
}
object Main extends App {
  val m = Model("data/model.onnx")
  println(m.predict_and_rank(Seq(5L, 20L)).mkString(" "))
}