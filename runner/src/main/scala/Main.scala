package server

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import ai.onnxruntime.OrtSession.SessionOptions

import scala.jdk.CollectionConverters._
import scala.util.Random

case class Model(path: String) {
  // If a mapping file named ${argName}.json exists in archive, use it. Else pass data through.
  // An archive should be a .zip file consisting of 1 onnx file, and optionally several json-files mapping inputs and outputs.
  private val env = OrtEnvironment.getEnvironment
  val session: OrtSession = env.createSession(path, new SessionOptions())
  private val argNames = session.getInputNames.asScala.toSeq
  println(f"Model argnames: ${argNames.mkString(", ")}") // Currently contentId, size

  // at load_time: make a Map[String, Map[T, Y]] or something that we can lookup in `map_in`
  val outputMap: Map[Long, String] = (0L until 10000L).map(i => i -> i.toString).toMap

  def map_in(values: Seq[Long]) = {
    argNames.zip(values).map(x => x._1 -> OnnxTensor.createTensor(env, scala.Array(x._2))).toMap.asJava
  }

  //TODO these things need to kinda be interfaces or contracts somehow.
  def map_out(output: OrtSession.Result) = {
    val scores = output.get(0).getValue.asInstanceOf[Array[Float]]
    val indices = output.get(1).getValue.asInstanceOf[Array[Long]]
    indices.zip(scores).map(x => outputMap.get(x._1) -> x._2) // todo collapse option, only return relevant.
  }
  def predict_and_rank(values: Seq[Long]) = {
    //val out = session.run(args(values))
    //out.get(0).getValue.asInstanceOf[Array[Float]]
    map_out(session.run(map_in(values)))
  }
}
object Main extends App {
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

  val m = Model("data/model.onnx")
  println(m.predict_and_rank(Seq(5L, 20L)).mkString(" "))

  // Just test some timings
  //1 to 10 map{_ => time(m.predict_and_rank(Seq(Random.between(0, 100), Random.between(5, 10))))}
}