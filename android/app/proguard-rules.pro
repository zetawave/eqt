# JNI entry points and the callback interface invoked from native code by name.
-keep class org.equity.app.NativeBridge { *; }
-keep interface org.equity.app.ByteSink { *; }
-keep class org.equity.app.ContextOverflowException { *; }
# AndroidMath: its FreeType JNI library resolves these classes and members by name.
-keep class com.pvporbit.freetype.** { *; }
-keep class com.agog.mathdisplay.** { *; }
