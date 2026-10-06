package org.equity.app

fun interface ByteSink { fun onBytes(bytes: ByteArray) }

object NativeBridge {
    init { System.loadLibrary("eqt_jni") }
    external fun prepare()
    external fun cancel()
    external fun unload()
    external fun load(path: ByteArray, options: ByteArray, callback: ByteSink): ByteArray
    external fun generate(request: ByteArray, callback: ByteSink): ByteArray
}
