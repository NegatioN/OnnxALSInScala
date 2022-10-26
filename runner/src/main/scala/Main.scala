package server

import ai.onnxruntime.{OnnxTensor, OrtEnvironment}
import ai.onnxruntime.OrtSession.SessionOptions

import java.util.Collections

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
  val inputNodeName = session.getInputNames.iterator.next
  val inp = OnnxTensor.createTensor(env, Array(5L));
  val inpSingleton = Collections.singletonMap(inputNodeName, inp)

  // Just test some timings
  val n = 100
  1 to n map{_ => time(session.run(inpSingleton))}

  // do full inference and output
  val out = session.run(inpSingleton)
  val actualOutput: Array[Float] = out.get(0).getValue.asInstanceOf[Array[Float]]
  println(actualOutput.mkString(" "))
}