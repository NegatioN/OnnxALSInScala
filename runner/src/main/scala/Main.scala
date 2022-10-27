package server

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import ai.onnxruntime.OrtSession.SessionOptions
import scala.jdk.CollectionConverters._

case class Model(path: String) {
  private val env = OrtEnvironment.getEnvironment
  val session: OrtSession = env.createSession(path, new SessionOptions())
  private val argNames = session.getInputNames.asScala.toSeq
  println(f"Model argnames: ${argNames.mkString(", ")}") // Currently contentId, size

  def map_in(values: Seq[Any]) = {
    //TODO more types would need to be handled, and probably arrays as well further down the line.
    val tensors = values.map {
      case v: Long => OnnxTensor.createTensor(env, Array(v))
      case v: String => OnnxTensor.createTensor(env, Array(v))
      case _ => OnnxTensor.createTensor(env, Array())
    }
    argNames.zip(tensors).toMap.asJava
  }

  def map_out(output: OrtSession.Result) = {
    // this code does not handle non-existent outputs very nicely :)
    val names = output.get("contentIdd").get().getValue.asInstanceOf[Array[String]]
    val scores = output.get("scores").get().getValue.asInstanceOf[Array[Double]]
    names.zip(scores)
  }
  def predict_and_rank(values: Seq[Any]) = {
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

  val m = Model("model.onnx")
  val input = Seq(10L, "lunsj")
  println(m.predict_and_rank(input).mkString(" "))

  // Just test some timings
  //1 to 10 map{_ => time(m.predict_and_rank(input))}
}