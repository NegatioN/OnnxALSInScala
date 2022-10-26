package server

import ai.onnxruntime.{OnnxTensor, OrtEnvironment}
import ai.onnxruntime.OrtSession.SessionOptions

import scala.collection.mutable
import scala.jdk.CollectionConverters.{MapHasAsJava, SetHasAsScala}
import scala.util.Random

object Main extends App {

  //todo make average time output (?)
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

  // Load model
  val env = OrtEnvironment.getEnvironment
  val opts = new SessionOptions()
  val session = env.createSession("data/model.onnx", opts)

  // create inputs
  def onnxargs(argNames: mutable.Set[String], values: Seq[Long])= {
    argNames.zip(values).map(x => x._1 -> OnnxTensor.createTensor(env, scala.Array(x._2))).toMap.asJava
  }
  val inputNodeNames = session.getInputNames.asScala
  val inp = onnxargs(inputNodeNames, Seq(5L, 10L)) // Seq[Any] fail... for some reason.

  // Just test some timings
  val n = 10
  1 to n map{_ => time(session.run(onnxargs(inputNodeNames, Seq(Random.between(0, 100), Random.between(5, 10)))))}

  // do full inference and output
  val out = session.run(inp)
  val actualOutput: Array[Float] = out.get(0).getValue.asInstanceOf[Array[Float]]
  println(actualOutput.mkString(" "))
}