#include "engine.h"
#include <jni.h>

namespace {
eqt::Engine &runtime() {
    static eqt::Engine engine;
    return engine;
}
std::string utf8(JNIEnv *env, jbyteArray bytes) {
    const auto size = env->GetArrayLength(bytes);
    std::string value(size, '\0');
    env->GetByteArrayRegion(bytes, 0, size, reinterpret_cast<jbyte *>(value.data()));
    return value;
}
jbyteArray bytes(JNIEnv *env, const std::string &value) {
    auto result = env->NewByteArray(static_cast<jsize>(value.size()));
    if (result) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(value.size()),
                                reinterpret_cast<const jbyte *>(value.data()));
    }
    return result;
}
void error(JNIEnv *env, const std::exception &e) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
}
} // namespace

extern "C" JNIEXPORT void JNICALL Java_org_equity_app_NativeBridge_prepare(JNIEnv *, jobject) {
    runtime().prepare();
}
extern "C" JNIEXPORT void JNICALL Java_org_equity_app_NativeBridge_cancel(JNIEnv *, jobject) {
    runtime().cancel();
}
extern "C" JNIEXPORT void JNICALL Java_org_equity_app_NativeBridge_unload(JNIEnv *, jobject) {
    runtime().unload();
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_org_equity_app_NativeBridge_load(JNIEnv *env, jobject,
                                                                              jbyteArray path,
                                                                              jbyteArray options,
                                                                              jobject callback) {
    try {
        const auto method = env->GetMethodID(env->GetObjectClass(callback), "onBytes", "([B)V");
        const auto result =
            runtime().load(utf8(env, path), eqt::Json::parse(utf8(env, options)), [&](const std::string &s) {
                auto data = bytes(env, s);
                if (data) {
                    env->CallVoidMethod(callback, method, data);
                    env->DeleteLocalRef(data);
                }
                if (env->ExceptionCheck()) {
                    runtime().cancel();
                }
            });
        return bytes(env, result.dump());
    } catch (const std::exception &e) {
        if (!env->ExceptionCheck()) {
            error(env, e);
        }
        return nullptr;
    }
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_org_equity_app_NativeBridge_generate(JNIEnv *env, jobject,
                                                                                  jbyteArray request,
                                                                                  jobject callback) {
    try {
        const auto method = env->GetMethodID(env->GetObjectClass(callback), "onBytes", "([B)V");
        const auto result =
            runtime().generate(eqt::Json::parse(utf8(env, request)), [&](const std::string &s) {
                auto data = bytes(env, s);
                if (data) {
                    env->CallVoidMethod(callback, method, data);
                    env->DeleteLocalRef(data);
                }
                if (env->ExceptionCheck()) {
                    runtime().cancel();
                }
            });
        return bytes(env, result.dump(-1, ' ', false, eqt::Json::error_handler_t::replace));
    } catch (const std::exception &e) {
        if (!env->ExceptionCheck()) {
            error(env, e);
        }
        return nullptr;
    }
}
