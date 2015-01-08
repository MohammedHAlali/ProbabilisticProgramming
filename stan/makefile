CMDSTAN=/home/abergman/apps/stan/cmdstan
STANC=${CMDSTAN}/bin/stanc
PRINT=${CMDSTAN}/bin/print
STANLINK=g++ \
		 -DBOOST_RESULT_OF_USE_TR1 -DBOOST_NO_DECLTYPE -DBOOST_DISABLE_ASSERTS \
		 -I ${CMDSTAN}/src \
		 -I ${CMDSTAN}/stan/src \
		 -isystem ${CMDSTAN}/stan/lib/eigen_3.2.0 \
		 -isystem ${CMDSTAN}/stan/lib/boost_1.54.0 \
		 -Wall -pipe -DEIGEN_NO_DEBUG  -lpthread
O=3
CMDSTAN_MAIN=${CMDSTAN}/src/cmdstan/main.cpp 
LDLIBS=-L${CMDSTAN}/bin -lstan

all: ./Simple_harmonic_oscillator-Estimator.x

%.cpp : %.stan
	@echo ''
	@echo '--- Translating Stan model to C++ code ---'
	${STANC} $< --o=$@ --no_main

.PRECIOUS: %.cpp
%.x : %.cpp 
	@echo ''
	@echo '--- Linking C++ model ---'
	$(STANLINK) -O$O $(OUTPUT_OPTION) $(CMDSTAN_MAIN) -include $< $(LDLIBS)

clean:
	rm -rfv *.x *.cpp .RData
