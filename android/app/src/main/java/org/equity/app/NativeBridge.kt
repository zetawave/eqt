package org.equity.app

fun interface ByteSink { fun onBytes(bytes: ByteArray) }

class ContextOverflowException(val promptTokens: Int, val outputTokens: Int, val contextTokens: Int) :
    IllegalStateException("Prompt plus output exceeds context: the message needs $promptTokens input tokens and $outputTokens output tokens, " +
        "but the loaded context holds $contextTokens. Increase the context or use a smaller attachment.")

object NativeBridge {
    init { System.loadLibrary("eqt_jni") }
    external fun prepare()
    external fun cancel()
    external fun unload()
    external fun load(path: ByteArray, options: ByteArray, callback: ByteSink): ByteArray
    external fun generate(request: ByteArray, callback: ByteSink): ByteArray
}
